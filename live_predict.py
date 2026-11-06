
"""Live webcam sign prediction for SignSync AI.

    python live_predict.py                # any language
    python live_predict.py --lang psl     # psl | asl | both (default both)
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


SEQ_LEN = 60
PREDICT_EVERY = 10
SMOOTHING = 3
STABLE_PREDICTIONS = 3
CONFIDENCE_THRESHOLD = 0.55


def predict_probabilities(sequence, session, names, lang="both"):
    # Same steps as training: trim empty start/end, then normalize.
    x = normalize(trim_active(np.asarray(sequence, dtype=np.float32)))[None].astype(np.float32)
    logits = session.run(None, {"landmarks": x})[0][0]
    logits = logits - logits.max()
    probabilities = np.exp(logits)
    return restrict_lang(probabilities / probabilities.sum(), names, lang)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["psl", "asl", "both"], default="both",
                        help="which sign language may be predicted (same single model)")
    args = parser.parse_args()

    # Load the trained ONNX model and class labels.
    session = ort.InferenceSession("sign_gru.onnx")
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
            landmarks = frame_to_vector(results)

            frames.append(landmarks)
            frame_count += 1

            hands_detected = (
                np.abs(landmarks).sum() > 0
            )

            # Predict only after collecting a complete 60-frame sequence.
            if len(frames) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
                recent = np.asarray(frames, dtype=np.float32)

                # Avoid predicting when hands are missing in most frames.
                present = np.abs(recent).sum(axis=1) > 0

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
                        and displayed_confidence >= CONFIDENCE_THRESHOLD
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

            if len(frames) == SEQ_LEN:
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
            if key == ord("s"):
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
