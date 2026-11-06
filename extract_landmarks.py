"""Step 3: extract hand landmarks from a sign video in EXACTLY the dataset's format,
so new PSL videos (e.g. from psl.org.pk), ASL videos, and live app input all go through
the same pipeline as PakistanSignLanguageDatasetV2.

Output per video: a folder with 0.npy ... 59.npy, each (126,) = [left hand 63 | right hand 63],
missing hand = zeros. Also prints shape + missing-hand stats (roadmap Step 3 check).

Usage:
    python extract_landmarks.py --video hello.mp4 --out out/hello/0
    python extract_landmarks.py --video-dir videos/ --out out/        # videos/<word>/*.mp4
    python extract_landmarks.py --webcam --out out/hello/0             # record 60 frames live

Needs: pip install "mediapipe==0.10.14" opencv-python numpy   (use Python 3.10 or 3.11)

Hand order: the dataset is consistent with MediaPipe Holistic's [left_hand, right_hand]
(the signer's own left/right, image NOT mirrored): slot 1 (right hand) is filled in most
frames, slot 0 only when both hands are used. Sanity-check with --verify-against <a dataset
sequence folder> on a right-handed sign: the dominant hand should land in the same slot.
"""
import argparse
import os
import sys
import numpy as np
import cv2

SEQ_LEN = 60


def _holistic():
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        sys.exit("This mediapipe build has no legacy 'solutions' API. Install: pip install 'mediapipe==0.10.14'")
    return mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5)


def frame_to_vector(results):
    def hand(lms):
        if lms is None:
            return np.zeros(63, dtype=np.float64)
        return np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float64).flatten()
    return np.concatenate([hand(results.left_hand_landmarks), hand(results.right_hand_landmarks)])


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def extract_from_frames(frames, model):
    """Run Holistic on EVERY frame (tracking needs continuity), then resample to 60 frames."""
    vecs = []
    for f in frames:
        res = model.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        vecs.append(frame_to_vector(res))
    vecs = np.array(vecs)
    if len(vecs) == 0:
        raise ValueError("no frames read")
    idx = np.linspace(0, len(vecs) - 1, SEQ_LEN).round().astype(int)
    return vecs[idx]


def save_sequence(seq, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, v in enumerate(seq):
        np.save(os.path.join(out_dir, f"{i}.npy"), v)


def report(seq, name):
    s = seq.reshape(len(seq), 2, 63)
    present = np.abs(s).sum(axis=2) > 0
    print(f"{name}: shape {seq.shape} | left hand in {present[:, 0].mean():.0%} of frames, "
          f"right hand in {present[:, 1].mean():.0%}, no hand in {(~present.any(1)).mean():.0%}")
    if (~present.any(1)).mean() > 0.5:
        print("   WARNING: hands missing in most frames - check lighting/framing or the video itself")


def record_webcam(n=SEQ_LEN, cam=0):
    cap = cv2.VideoCapture(cam)
    frames = []
    print("Recording starts in 2s - perform the sign...")
    for _ in range(60):  # ~2s countdown
        cap.read()
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        cv2.imshow("recording (q to stop)", f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frames


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--video-dir", help="folder of <word>/<file>.mp4")
    g.add_argument("--webcam", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-against", help="a dataset sequence folder to compare hand slots with")
    a = ap.parse_args()

    model = _holistic()
    if a.video or a.webcam:
        frames = read_frames(a.video) if a.video else record_webcam()
        seq = extract_from_frames(frames, model)
        save_sequence(seq, a.out)
        report(seq, a.video or "webcam")
    else:
        exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
        for word in sorted(os.listdir(a.video_dir)):
            wdir = os.path.join(a.video_dir, word)
            if not os.path.isdir(wdir):
                continue
            vids = sorted(f for f in os.listdir(wdir) if f.lower().endswith(exts))
            for i, v in enumerate(vids):
                try:
                    seq = extract_from_frames(read_frames(os.path.join(wdir, v)), model)
                except ValueError as e:
                    print(f"skip {word}/{v}: {e}")
                    continue
                save_sequence(seq, os.path.join(a.out, word, str(i)))
                report(seq, f"{word}/{v}")

    if a.verify_against:
        ref = np.stack([np.load(os.path.join(a.verify_against, f"{i}.npy")) for i in range(SEQ_LEN)])
        report(ref, "dataset reference")
        print("If the dominant hand is in the opposite slot from the reference, swap halves: "
              "seq = np.concatenate([seq[:, 63:], seq[:, :63]], axis=1)")


if __name__ == "__main__":
    main()
