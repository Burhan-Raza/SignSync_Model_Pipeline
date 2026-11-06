"""Step 5: train a GRU sign classifier on the prepared landmarks.

Usage (laptop or Google Colab):
    pip install torch numpy scikit-learn onnx
    python train_gru.py --data data_prepared.npz --epochs 80

Saves: gru_best.pt (weights), labels.json (index -> word), sign_gru.onnx (for FastAPI/onnxruntime).
Prints val accuracy each epoch and final test accuracy + most-confused pairs.

Trains from X_raw, the same way baseline_sklearn.py does: every training batch gets fresh
common.augment_raw + random common.mirror, then common.trim_active + common.normalize.
Val/test get trim_active + normalize only, plus a "stress" (live-like) test with augment + mirror.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from common import normalize, trim_active, mirror, augment_raw, lang_accuracy, cross_language_confusions


class SignGRU(nn.Module):
    def __init__(self, n_feat, n_classes, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def forward(self, x):                       # x: (B, 60, F)
        h, _ = self.gru(self.inp(x))           # (B, 60, 2H)
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=1)
        return self.head(pooled)


def prep_raw(raws):
    """Raw (N, 60, 126) -> model input (N, 60, 130): same steps as live prediction."""
    return np.stack([normalize(trim_active(r)) for r in raws]).astype(np.float32)


def augment_raw_batch(raws, rng, mirror_p=0.5):
    """Random live-like copy of each raw sequence (half of them mirrored), ready for the model."""
    out = []
    for r in raws:
        v = augment_raw(r, rng)
        out.append(mirror(v) if rng.random() < mirror_p else v)
    return prep_raw(out)


def export_onnx(model, n_frames, n_feat, path="sign_gru.onnx"):
    """Export with a batch of 2 and a named batch dimension, so the ONNX model accepts any batch size."""
    torch.onnx.export(model, torch.zeros(2, n_frames, n_feat), path,
                      input_names=["landmarks"], output_names=["logits"], verbose=False,
                      dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}})


def print_lang_accuracy(title, p, y, words):
    print(f"{title}: " + "  ".join(f"{k} {v:.3f}" for k, v in lang_accuracy(y, p, words).items()))


def evaluate(model, loader):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).argmax(1))
            ys.append(yb)
    p, y = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return (p == y).mean(), p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_prepared.npz")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    d = np.load(a.data)
    if "X_raw" not in d:
        raise SystemExit("This .npz was made by the old prepare.py - re-run prepare.py to get X_raw.")
    R, y, split, words = d["X_raw"].astype(np.float32), d["y"].astype(np.int64), d["split"], d["words"]
    rng = np.random.default_rng(0)
    mk = lambda x, s, sh: DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y[split == s])),
                                     batch_size=a.batch, shuffle=sh)
    tr = mk(R[split == "train"], "train", True)              # raw: prepared fresh for every batch
    tr_clean = mk(prep_raw(R[split == "train"]), "train", False)   # only to report training accuracy
    va, te = mk(prep_raw(R[split == "val"]), "val", False), mk(prep_raw(R[split == "test"]), "test", False)
    # same stress recipe (and seed) as baseline_sklearn.py
    va_stress = mk(augment_raw_batch(R[split == "val"], np.random.default_rng(1)), "val", False)
    te_stress = mk(augment_raw_batch(R[split == "test"], np.random.default_rng(1)), "test", False)
    print(f"{len(words)} words: {list(words)}")
    print(f"train {len(tr.dataset)}  val {len(va.dataset)}  test {len(te.dataset)}", flush=True)

    n_feat = va.dataset[0][0].shape[1]
    model = SignGRU(n_feat, len(words))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, bad = 0.0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for rb, yb in tr:
            xb = torch.from_numpy(augment_raw_batch(rb.numpy(), rng))
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        acc, _, _ = evaluate(model, va)
        tr_acc, _, _ = evaluate(model, tr_clean)
        print(f"epoch {ep:3d}  loss {tot / len(tr.dataset):.3f}  train acc {tr_acc:.3f}  val acc {acc:.3f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            torch.save(model.state_dict(), "gru_best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    model.load_state_dict(torch.load("gru_best.pt"))
    acc, p, t = evaluate(model, te)
    print(f"\nbest val acc {best:.3f} | TEST acc {acc:.3f}")
    _, pt, tt = evaluate(model, tr_clean)
    print_lang_accuracy("clean train by language", pt, tt, words)
    _, pv0, tv0 = evaluate(model, va)
    print_lang_accuracy("clean val   by language", pv0, tv0, words)
    print_lang_accuracy("clean test  by language", p, t, words)
    _, ps, ts = evaluate(model, te_stress)
    print_lang_accuracy("STRESS test by language", ps, ts, words)
    _, pv, tv = evaluate(model, va_stress)
    print_lang_accuracy("STRESS val  by language", pv, tv, words)
    cross = cross_language_confusions(t, p, words)
    print("confused across languages (clean test): " + (", ".join(f"{a} -> {b} x{n}" for a, b, n in cross) or "none"))
    cm = np.zeros((len(words), len(words)), int)
    np.add.at(cm, (t, p), 1)
    np.fill_diagonal(cm, 0)
    for i, j in zip(*np.unravel_index(np.argsort(cm, axis=None)[::-1][:8], cm.shape)):
        if cm[i, j]:
            print(f"  {words[i]:>15s} -> predicted {words[j]:<15s} x{cm[i, j]}")

    json.dump({int(i): str(w) for i, w in enumerate(words)}, open("labels.json", "w"), indent=1)
    model.eval()
    export_onnx(model, R.shape[1], n_feat)
    print("saved gru_best.pt, labels.json, sign_gru.onnx")


if __name__ == "__main__":
    main()
