"""Live / file prediction with the scikit-learn baseline (baseline_model.joblib).

The input goes through EXACTLY the same steps as the training data:
    MediaPipe Holistic on the UN-mirrored frame -> (T,126) -> fill short gaps -> trim empty
    start/end -> resample to 60 frames -> normalize -> features -> model.

Modes
    python live_baseline.py                      # SPACE = record one sign, q = quit  (recommended)
    python live_baseline.py --mode auto          # starts when hands appear, predicts when they drop
    python live_baseline.py --video my_sign.mp4  # one video file
    python live_baseline.py --seq-dir out/hello/0
    python live_baseline.py --lang psl           # psl | asl | both (default both), any mode above
    python live_baseline.py --collect excuseme --save-dir my_data   # record YOUR OWN samples
        (SPACE records one sample and saves it as my_data/excuseme/<n>/0..59.npy; then run
         prepare.py --extra-root my_data so your own signing is in the training set)

Tips: put your hands DOWN (out of frame) between signs - that is how the start/end of a sign
is found. Show the whole sign from start to finish, at normal speed.
"""
import argparse
import collections
import os
import sys
import time
import numpy as np
import joblib
import cv2
from common import normalize, trim_active, hand_presence, SEQ_LEN, restrict_lang, strip_lang
from baseline_sklearn import featurize_one


def load_model(path="baseline_model.joblib"):
    b = joblib.load(path)
    return b["model"], b["words"]


def predict_raw(raw, model, words, top=3, lang="both"):
    """raw: (T,126) landmarks in dataset format, any length T >= 5 -> [(word, prob), ...]
    lang psl/asl: only that language's classes can win. Words come back without the prefix."""
    raw = np.asarray(raw, dtype=np.float32)
    x = featurize_one(normalize(trim_active(raw)))[None]
    p = model.predict_proba(x)[0]
    cls = model.classes_ if hasattr(model, "classes_") else np.arange(len(p))
    names = [words[int(c)] for c in cls]
    p = restrict_lang(p, names, lang)
    order = [i for i in np.argsort(p)[::-1] if p[i] > 0][:top]
    return [(strip_lang(names[i]), float(p[i])) for i in order]


def describe(res, threshold):
    (w, p), rest = res[0], res[1:]
    head = f"{w} ({p:.0%})" if p >= threshold else f"? not sure - maybe {w} ({p:.0%})"
    return head + "   | " + ", ".join(f"{a} {b:.0%}" for a, b in rest)


def _holistic():
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        sys.exit("This mediapipe build has no legacy 'solutions' API. Install: pip install 'mediapipe==0.10.14'")
    return mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5)


def frame_vec(frame_bgr, holo, input_is_mirrored):
    """MediaPipe must see the NON-mirrored image (that's how the dataset was made)."""
    if input_is_mirrored:
        frame_bgr = cv2.flip(frame_bgr, 1)
    res = holo.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def hand(lms):
        if lms is None:
            return np.zeros(63, np.float32)
        return np.array([[q.x, q.y, q.z] for q in lms.landmark], np.float32).ravel()
    return np.concatenate([hand(res.left_hand_landmarks), hand(res.right_hand_landmarks)])


def save_seq(raw, out_dir):
    seq = trim_active(raw)
    os.makedirs(out_dir, exist_ok=True)
    for i, v in enumerate(seq):
        np.save(os.path.join(out_dir, f"{i}.npy"), v.astype(np.float64))


def run_camera(a, model, words):
    holo = _holistic()
    cap = cv2.VideoCapture(a.cam)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_rec = int(round(a.seconds * (fps if 5 < fps < 121 else 30)))
    pre = collections.deque(maxlen=6)          # a few frames before the start (auto mode)
    buf, state, last, miss = [], "idle", "", 0
    t_countdown = 0.0
    n_saved = 0
    if a.collect:
        wdir = os.path.join(a.save_dir, a.collect)
        os.makedirs(wdir, exist_ok=True)
        n_saved = len([x for x in os.listdir(wdir) if x.isdigit()])
    print("SPACE = record a sign, q = quit" if a.mode == "space" else "auto mode: raise hands to sign, drop them to finish; q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        v = frame_vec(frame, holo, a.input_mirrored)
        has_hand = np.abs(v).sum() > 0
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if a.mode == "space":
            if state == "idle" and key == ord(" "):
                state, t_countdown, buf = "countdown", time.time(), []
            elif state == "countdown" and time.time() - t_countdown > a.countdown:
                state = "rec"
            elif state == "rec":
                buf.append(v)
                if len(buf) >= n_rec:
                    state = "done"
        else:  # auto
            if state == "idle":
                pre.append(v)
                miss = 0 if has_hand else miss
                if has_hand and sum(np.abs(x).sum() > 0 for x in list(pre)[-3:]) >= 3:
                    state, buf = "rec", list(pre)
            elif state == "rec":
                buf.append(v)
                miss = 0 if has_hand else miss + 1
                if miss >= a.end_frames or len(buf) >= 4 * SEQ_LEN:
                    state = "done"

        if state == "done":
            raw = np.array(buf, np.float32)
            frac = hand_presence(raw).any(1).mean()
            if frac < 0.2 or len(raw) < 10:
                last = "no hands seen - try again (check light / framing)"
            elif a.collect:
                save_seq(raw, os.path.join(a.save_dir, a.collect, str(n_saved)))
                n_saved += 1
                last = f"saved {a.collect} #{n_saved}"
            else:
                last = describe(predict_raw(raw, model, words, lang=a.lang), a.threshold)
            print(last)
            state, buf, miss = "idle", [], 0
            pre.clear()

        view = cv2.flip(frame, 1) if not a.input_mirrored else frame.copy()   # selfie view for display only
        status = {"idle": "ready (SPACE)" if a.mode == "space" else "ready - raise hands",
                  "countdown": "get ready...", "rec": f"RECORDING {len(buf)}"}.get(state, "")
        cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if state == "rec" else (0, 200, 0), 2)
        cv2.putText(view, last[:60], (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("SignSync baseline", view)
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline_model.joblib")
    ap.add_argument("--video")
    ap.add_argument("--seq-dir")
    ap.add_argument("--mode", choices=["space", "auto"], default="space")
    ap.add_argument("--seconds", type=float, default=2.5, help="recording length in SPACE mode")
    ap.add_argument("--countdown", type=float, default=0.7)
    ap.add_argument("--end-frames", type=int, default=8, help="auto mode: frames without hands that end a sign")
    ap.add_argument("--threshold", type=float, default=0.45, help="below this the answer is shown as 'not sure'")
    ap.add_argument("--input-mirrored", action="store_true",
                    help="set this if your camera driver already gives a mirrored (selfie) image")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--collect", help="word to record samples for (your own training data)")
    ap.add_argument("--save-dir", default="my_data")
    ap.add_argument("--lang", choices=["psl", "asl", "both"], default="both",
                    help="which sign language may be predicted (same single model)")
    a = ap.parse_args()

    model, words = (None, None) if a.collect else load_model(a.model)
    if words:
        restrict_lang(np.ones(len(words)), words, a.lang)   # wrong --lang: fail now, not after recording
    if a.seq_dir:
        files = sorted((f for f in os.listdir(a.seq_dir) if f.endswith(".npy")), key=lambda f: int(f[:-4]))
        raw = np.stack([np.load(os.path.join(a.seq_dir, f)) for f in files])
        print(describe(predict_raw(raw, model, words, lang=a.lang), a.threshold))
    elif a.video:
        holo, cap, vecs = _holistic(), cv2.VideoCapture(a.video), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            vecs.append(frame_vec(f, holo, a.input_mirrored))
        print(describe(predict_raw(np.array(vecs), model, words, lang=a.lang), a.threshold))
    else:
        run_camera(a, model, words)


if __name__ == "__main__":
    main()
