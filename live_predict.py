
"""Live webcam sign prediction for SignSync AI.

    python live_predict.py                # any language
    python live_predict.py --lang psl     # psl | asl | both (default both)
    python live_predict.py --mode window  # old behaviour: a new guess every 10 frames

    python live_predict.py --hold 2       # keep the answer on screen 2 s after the hands leave

Loads sign_conformer.onnx from the current folder (--model picks a file).

Default mode "sign": raise your hands and sign. The word appears while your hands are in the frame
and disappears when you take them out of the frame. Each sign is read from its start, so holding
a pose still does not change the answer (after about 3 s it is fixed).
"""

import argparse
import cv2
import json
import time
import numpy as np
import onnxruntime as ort
from collections import deque

from common import normalize, trim_active, restrict_lang, strip_lang
from extract_landmarks import _holistic, frame_to_vector
from predict import default_model_path
from body import frame_to_body_vector, normalize_body, trim_body, FEAT_DIM


SEQ_LEN = 60
PREDICT_EVERY = 10
SMOOTHING = 3
STABLE_PREDICTIONS = 3
CONFIDENCE_THRESHOLD = 0.50   # answers below this confidence are not shown (both modes; see --threshold)

# "sign" mode: when a sign starts and ends.
START_FRAMES = 3        # hands seen in this many frames in a row -> a sign starts
END_FRAMES = 12         # hands gone for this many frames (about 0.4 s) -> the sign is over
PRE_FRAMES = 6          # frames kept from just before the start
MAX_SIGN_FRAMES = 90    # frames kept per sign (about 3 s); holding the pose longer adds nothing
MIN_HAND_FRAMES = 10    # shorter bits (e.g. after MediaPipe briefly loses a hand) are ignored
LIVE_MIN_FRAMES = 15    # hand frames needed before the first live answer is shown
LIVE_EVERY = 5          # while signing, refresh the answer every this many frames
HIDE_FRAMES = 5         # hands gone for this many frames -> the word leaves the screen


def read_train_aspect(path="input_aspect.txt"):
    """Width/height of the training videos, if the model folder says so (first number in the file)."""
    try:
        with open(path, encoding="utf-8") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def match_aspect(landmarks, factor):
    """Stretch x (and z) around the image centre by factor; missing points (all zero) stay zero."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3).copy()
    ok = np.abs(pts).sum(axis=1) > 0
    pts[ok, 0] = 0.5 + (pts[ok, 0] - 0.5) * factor
    pts[ok, 2] = pts[ok, 2] * factor
    return pts.ravel()


def predict_probabilities(sequence, session, names, lang="both"):
    # Same steps as training: trim empty start/end, then normalize. 165 numbers per frame = hands + face/body.
    sequence = np.asarray(sequence, dtype=np.float32)
    feats = normalize_body(trim_body(sequence)) if sequence.shape[1] > 126 else normalize(trim_active(sequence))
    x = feats[None].astype(np.float32)
    logits = session.run(None, {"landmarks": x})[0][0]
    logits = logits - logits.max()
    probabilities = np.exp(logits)
    return restrict_lang(probabilities / probabilities.sum(), names, lang)


class SignSegmenter:
    """Cuts the webcam stream into single signs: a sign starts when hands appear and
    ends when they have been gone for END_FRAMES frames. push() returns the finished
    sign's frames once, or None while nothing has finished."""

    def __init__(self, start_frames=START_FRAMES, end_frames=END_FRAMES,
                 pre_frames=PRE_FRAMES, max_frames=MAX_SIGN_FRAMES, min_hand_frames=MIN_HAND_FRAMES):
        self.start_frames, self.end_frames, self.max_frames = start_frames, end_frames, max_frames
        self.min_hand_frames = min_hand_frames
        self.pre = deque(maxlen=max(pre_frames, start_frames))
        self.buf, self.miss, self.recording = [], 0, False

    @property
    def full(self):
        return self.recording and len(self.buf) >= self.max_frames

    def push(self, landmarks):
        has_hand = np.abs(landmarks[:126]).sum() > 0          # only the hands start and end a sign
        if not self.recording:
            self.pre.append(landmarks)
            recent = list(self.pre)[-self.start_frames:]
            if len(recent) == self.start_frames and all(np.abs(v[:126]).sum() > 0 for v in recent):
                self.recording, self.buf, self.miss = True, list(self.pre), 0
            return None
        if len(self.buf) < self.max_frames:
            self.buf.append(landmarks)
        self.miss = 0 if has_hand else self.miss + 1
        if self.miss < self.end_frames:
            return None
        sign = np.asarray(self.buf, dtype=np.float32)
        self.recording, self.buf, self.miss = False, [], 0
        self.pre.clear()
        hands = np.abs(sign[:, :126]).sum(axis=1) > 0
        return sign if hands.sum() >= self.min_hand_frames and hands.mean() >= 0.2 else None


class LiveSign:
    """"sign" mode without the camera, so it can be tested on recorded data.
    While hands are in the frame it keeps an answer for the sign so far (always read from the
    sign's start); `visible` turns False as soon as the hands have left the frame."""

    def __init__(self, classify):
        self.classify = classify                 # (T, 126) array -> (word, confidence)
        self.seg = SignSegmenter()
        self.word, self.conf, self._count, self._frozen = "", 0.0, 0, False

    @property
    def visible(self):
        return self.seg.recording and self.seg.miss < HIDE_FRAMES and bool(self.word)

    def push(self, landmarks):
        """Feed one frame. Returns the finished sign's frames once when a sign ends, else None."""
        finished = self.seg.push(landmarks)
        if not self.seg.recording:
            self.word, self.conf, self._count, self._frozen = "", 0.0, 0, False
            return finished
        self._count += 1
        if not self._frozen and self._count % LIVE_EVERY == 0:
            buf = np.asarray(self.seg.buf, dtype=np.float32)
            if (np.abs(buf[:, :126]).sum(axis=1) > 0).sum() >= LIVE_MIN_FRAMES:
                self.word, self.conf = self.classify(buf)
                self._frozen = self.seg.full     # buffer is full: the answer cannot change any more
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["psl", "asl", "both"], default="both",
                        help="which sign language may be predicted (same single model)")
    parser.add_argument("--mode", choices=["sign", "window"], default="sign",
                        help="sign: one answer per sign (default); window: a new guess every 10 frames")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="hide answers below this confidence (default 0.50; 0 = always show)")
    parser.add_argument("--hold", type=float, default=0.0,
                        help="sign mode: seconds the answer stays on screen after the hands leave (default 0)")
    parser.add_argument("--model", help="ONNX file (default: sign_conformer.onnx in the current folder)")
    parser.add_argument("--no-aspect-fix", action="store_true",
                        help="do not rescale webcam x to the training videos' image shape (input_aspect.txt)")
    args = parser.parse_args()
    threshold = args.threshold

    # Load the trained ONNX model and class labels.
    model_path = args.model or default_model_path()
    session = ort.InferenceSession(model_path)
    uses_body = session.get_inputs()[0].shape[-1] == FEAT_DIM
    print(f"Model: {model_path}" + (" (hands + face/body points)" if uses_body else " (hands only)"))
    train_aspect = None if args.no_aspect_fix else read_train_aspect()
    aspect_factor = None                      # set from the first webcam frame
    with open("labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    names = [labels[str(i)] for i in range(len(labels))]
    restrict_lang(np.ones(len(names)), names, args.lang)   # wrong --lang: fail before opening the camera

    holistic = _holistic()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        holistic.close()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions or camera index."
        )

    frames = deque(maxlen=SEQ_LEN)
    probability_history = deque(maxlen=SMOOTHING)
    candidate_history = deque(maxlen=STABLE_PREDICTIONS)

    frame_count = 0
    displayed_word = "Show your hands to the camera"
    displayed_confidence = 0.0
    last_sign = None
    held = None                                  # (word, confidence, show-until time) for --hold

    def classify(sequence):
        probabilities = predict_probabilities(sequence, session, names, args.lang)
        best = int(np.argmax(probabilities))
        return strip_lang(names[best]), float(probabilities[best])

    reader = LiveSign(classify)

    if args.mode == "sign":
        displayed_word = ""
        print("Live SignSync prediction started. The word shows while your hands are in the frame and "
              "clears when they leave. S = save the last sign + top 3, Q = quit.")
    else:
        print("Live SignSync prediction started. Press S to save/print top predictions, Q to quit.")

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read a webcam frame.")
                break

            # Process the original, unmirrored frame to preserve
            # the dataset's left-hand/right-hand landmark convention.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            landmarks = frame_to_body_vector(results) if uses_body else frame_to_vector(results)
            if train_aspect:
                if aspect_factor is None:
                    aspect_factor = (frame.shape[1] / frame.shape[0]) / train_aspect
                    print(f"Image-shape fix: webcam {frame.shape[1]}x{frame.shape[0]} -> training videos "
                          f"(width/height {train_aspect:.2f}), x scaled by {aspect_factor:.2f}")
                landmarks = match_aspect(landmarks, aspect_factor)

            frames.append(landmarks)
            frame_count += 1

            hands_detected = (
                np.abs(landmarks[:126]).sum() > 0
            )

            if args.mode == "sign":
                finished = reader.push(landmarks)
                if finished is not None:                       # hands left: the sign is over
                    last_sign = finished
                    word, confidence = classify(finished)
                    print(f"{word}  ({confidence:.0%})" if confidence >= threshold
                          else f"(not shown: {word} {confidence:.0%} is below {threshold:.0%})")
                    held = (word, confidence, time.time() + args.hold) if args.hold > 0 and confidence >= threshold else None
                if reader.visible and reader.conf >= threshold:
                    displayed_word, displayed_confidence, held = reader.word, reader.conf, None
                elif held and time.time() < held[2]:
                    displayed_word, displayed_confidence = held[0], held[1]
                else:
                    displayed_word, displayed_confidence, held = "", 0.0, None

            # Predict only after collecting a complete 60-frame sequence.
            if args.mode == "window" and len(frames) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
                recent = np.asarray(frames, dtype=np.float32)

                # Avoid predicting when hands are missing in most frames.
                present = np.abs(recent[:, :126]).sum(axis=1) > 0

                if present.mean() >= 0.30:
                    probabilities = predict_probabilities(recent, session, names, args.lang)

                    probability_history.append(probabilities)

                    # Smooth predictions across recent windows.
                    averaged = np.mean(
                        np.stack(probability_history), axis=0
                    )

                    candidate_history.append(int(np.argmax(averaged)))
                    votes = np.bincount(
                        candidate_history, minlength=len(labels)
                    )
                    best_index = int(np.argmax(votes))
                    displayed_confidence = float(averaged[best_index])

                    if (
                        len(candidate_history) == STABLE_PREDICTIONS
                        and votes[best_index] >= 2
                        and displayed_confidence >= threshold
                    ):
                        displayed_word = strip_lang(names[best_index])
                    else:
                        displayed_word = "Recognizing sign..."
                else:
                    displayed_word = "Show both hands clearly"
                    displayed_confidence = 0.0
                    probability_history.clear()
                    candidate_history.clear()

            # Mirror only the display, not the processed input.
            display = cv2.flip(frame, 1)

            cv2.rectangle(display, (10, 10), (display.shape[1] - 10, 125),
                          (25, 25, 25), -1)

            cv2.putText(
                display, "SignSync AI - Live Recognition",
                (25, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

            cv2.putText(
                display, f"Sign: {displayed_word}",
                (25, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2
            )

            if args.mode == "sign":
                if displayed_word:
                    status = f"Confidence: {displayed_confidence:.0%}"
                elif reader.seg.recording and reader.seg.miss == 0:
                    status = "Hands seen - keep signing..."
                else:
                    status = "Ready - raise your hands"
            elif len(frames) == SEQ_LEN:
                status = f"Confidence: {displayed_confidence:.1%}"
            else:
                status = f"Collecting frames: {len(frames)}/{SEQ_LEN}"

            cv2.putText(
                display, status,
                (25, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

            cv2.putText(
                display, "Press S: capture + top 3 | Q: quit",
                (25, 117), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.imshow("SignSync AI - Live", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and args.mode == "sign":
                if last_sign is None:
                    print("Nothing to save yet: do a sign and lower your hands first.")
                else:
                    probabilities = predict_probabilities(last_sign, session, names, args.lang)
                    top_indices = [i for i in np.argsort(probabilities)[::-1] if probabilities[i] > 0][:3]
                    capture_path = f"live_capture_{time.strftime('%Y%m%d_%H%M%S')}.npy"
                    np.save(capture_path, last_sign)
                    print(f"\nSaved the last sign ({len(last_sign)} frames): {capture_path}")
                    print("Top predictions for this sign:")
                    for index in top_indices:
                        print(f"  {strip_lang(names[int(index)]):20s} {probabilities[index]:.1%}")
            elif key == ord("s"):
                if len(frames) < SEQ_LEN:
                    print(f"Capture unavailable: collect {SEQ_LEN} frames first.")
                else:
                    captured = np.asarray(frames, dtype=np.float32)
                    probabilities = predict_probabilities(captured, session, names, args.lang)
                    top_indices = [i for i in np.argsort(probabilities)[::-1] if probabilities[i] > 0][:3]
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    capture_path = f"live_capture_{timestamp}.npy"
                    np.save(capture_path, captured)
                    print(f"\nSaved exact 60-frame window: {capture_path}")
                    print("Top predictions for this window:")
                    for index in top_indices:
                        print(f"  {strip_lang(names[int(index)]):20s} {probabilities[index]:.1%}")
            elif key == ord("q"):
                break

    finally:
        camera.release()
        holistic.close()
        cv2.destroyAllWindows()
        print("Live prediction stopped.")


if __name__ == "__main__":
    main()
