"""Predict a sign from a video or a 60-frame .npy folder using the exported ONNX model.
This is the exact function your FastAPI endpoint will call.

    python predict.py --video my_sign.mp4
    python predict.py --seq-dir out/hello/0
    python predict.py --seq-dir out/hello/0 --lang psl      # psl | asl | both (default both)
"""
import argparse
import json
import os
import numpy as np
import onnxruntime as ort
from common import normalize, trim_active, restrict_lang, strip_lang
from body import normalize_body, trim_body, FEAT_DIM, RAW_DIM


def default_model_path(folder="."):
    """The model in this folder: the Conformer if there is one, otherwise the GRU."""
    conformer = os.path.join(folder, "sign_conformer.onnx")
    return conformer if os.path.exists(conformer) else os.path.join(folder, "sign_gru.onnx")


def load_model(onnx_path=None, labels_path="labels.json"):
    return ort.InferenceSession(onnx_path or default_model_path()), json.load(open(labels_path))


def predict(seq_126, sess, labels, top=3, lang="both"):
    """seq_126: raw (60, 126) landmarks in dataset format -> [(word, prob), ...]
    lang psl/asl: only that language's classes can win. Words come back without the prefix."""
    seq = np.asarray(seq_126, dtype=np.float32)
    if sess.get_inputs()[0].shape[-1] == FEAT_DIM:            # model with face/body points (body.py)
        if seq.shape[1] != RAW_DIM:
            raise ValueError(f"this model needs {RAW_DIM} numbers per frame (hands + face/body points), got {seq.shape[1]}")
        x = normalize_body(trim_body(seq))[None].astype(np.float32)
    else:
        x = normalize(trim_active(seq))[None].astype(np.float32)
    logits = sess.run(None, {"landmarks": x})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    names = [labels[str(i)] for i in range(len(p))]
    p = restrict_lang(p, names, lang)
    order = [i for i in np.argsort(p)[::-1] if p[i] > 0][:top]
    return [(strip_lang(names[i]), float(p[i])) for i in order]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video"); g.add_argument("--seq-dir")
    ap.add_argument("--lang", choices=["psl", "asl", "both"], default="both",
                    help="which sign language may be predicted (same single model)")
    ap.add_argument("--model", help="ONNX file (default: sign_conformer.onnx if present, else sign_gru.onnx)")
    a = ap.parse_args()
    if a.video:
        from extract_landmarks import _holistic, read_frames, extract_from_frames
        seq = extract_from_frames(read_frames(a.video), _holistic())
    else:
        seq = np.stack([np.load(os.path.join(a.seq_dir, f"{i}.npy")) for i in range(60)])
    sess, labels = load_model(a.model)
    for w, p in predict(seq, sess, labels, lang=a.lang):
        print(f"{w:20s} {p:.1%}")
