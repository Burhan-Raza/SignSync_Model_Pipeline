"""Predict a sign from a video or a 60-frame .npy folder using the exported ONNX model.
This is the exact function your FastAPI endpoint will call.

    python predict.py --video my_sign.mp4
    python predict.py --seq-dir out/hello/0
"""
import argparse
import json
import os
import numpy as np
import onnxruntime as ort
from common import normalize


def load_model(onnx_path="sign_gru.onnx", labels_path="labels.json"):
    return ort.InferenceSession(onnx_path), json.load(open(labels_path))


def predict(seq_126, sess, labels, top=3):
    """seq_126: raw (60, 126) landmarks in dataset format -> [(word, prob), ...]"""
    x = normalize(np.asarray(seq_126, dtype=np.float32))[None].astype(np.float32)
    logits = sess.run(None, {"landmarks": x})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    return [(labels[str(i)], float(p[i])) for i in np.argsort(p)[::-1][:top]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video"); g.add_argument("--seq-dir")
    a = ap.parse_args()
    if a.video:
        from extract_landmarks import _holistic, read_frames, extract_from_frames
        seq = extract_from_frames(read_frames(a.video), _holistic())
    else:
        seq = np.stack([np.load(os.path.join(a.seq_dir, f"{i}.npy")) for i in range(60)])
    sess, labels = load_model()
    for w, p in predict(seq, sess, labels):
        print(f"{w:20s} {p:.1%}")
