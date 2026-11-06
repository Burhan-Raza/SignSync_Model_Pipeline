"""Step 5: train a GRU sign classifier on the prepared landmarks.

Usage (laptop or Google Colab):
    pip install torch numpy scikit-learn onnx
    python train_gru.py --data data_prepared.npz --epochs 80

Saves: gru_best.pt (weights), labels.json (index -> word), sign_gru.onnx (for FastAPI/onnxruntime).
Prints val accuracy each epoch and final test accuracy + most-confused pairs.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def augment(x):
    """Light, sign-safe augmentation on normalized landmarks (B, T, F)."""
    B = x.shape[0]
    x = x * (1 + 0.1 * torch.randn(B, 1, 1))                 # global scale jitter
    x = x + 0.02 * torch.randn_like(x) * (x != 0)            # small noise on present points only
    # random speed change: resample time axis by factor 0.8-1.2
    T = x.shape[1]
    f = float(np.random.uniform(0.8, 1.2))
    idx = torch.clamp((torch.arange(T) * f).long(), max=T - 1)
    return x[:, idx]


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
    X, y, split, words = d["X"].astype(np.float32), d["y"].astype(np.int64), d["split"], d["words"]
    mk = lambda s, sh: DataLoader(TensorDataset(torch.from_numpy(X[split == s]), torch.from_numpy(y[split == s])),
                                  batch_size=a.batch, shuffle=sh)
    tr, va, te = mk("train", True), mk("val", False), mk("test", False)

    model = SignGRU(X.shape[2], len(words))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, bad = 0.0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr:
            opt.zero_grad()
            loss = lossf(model(augment(xb)), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        acc, _, _ = evaluate(model, va)
        print(f"epoch {ep:3d}  loss {tot / len(tr.dataset):.3f}  val acc {acc:.3f}")
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
    cm = np.zeros((len(words), len(words)), int)
    np.add.at(cm, (t, p), 1)
    np.fill_diagonal(cm, 0)
    for i, j in zip(*np.unravel_index(np.argsort(cm, axis=None)[::-1][:8], cm.shape)):
        if cm[i, j]:
            print(f"  {words[i]:>15s} -> predicted {words[j]:<15s} x{cm[i, j]}")

    json.dump({int(i): str(w) for i, w in enumerate(words)}, open("labels.json", "w"), indent=1)
    model.eval()
    torch.onnx.export(model, torch.zeros(1, X.shape[1], X.shape[2]), "sign_gru.onnx",
                      input_names=["landmarks"], output_names=["logits"],
                      dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}})
    print("saved gru_best.pt, labels.json, sign_gru.onnx")


if __name__ == "__main__":
    main()
