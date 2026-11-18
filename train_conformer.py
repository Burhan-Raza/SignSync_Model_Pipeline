"""Train a small Conformer sign classifier, to compare with the GRU (train_gru.py).

Idea from: Haque et al., "A Signer-Invariant Conformer and Multi-Scale Fusion Transformer for
Continuous Sign Language Recognition", ICCV 2025 Workshops. That paper does continuous (sentence)
recognition; this is a small Conformer encoder used as an ISOLATED sign classifier, so it is
inspired by the paper, not a reproduction of it.

Everything except the network is the same as train_gru.py: same .npz and split, training from X_raw
with augment_raw + random mirror -> trim_active + normalize, same stress test, same reports.

    python train_conformer.py --data psl_asl.npz --epochs 80

Saves: conformer_best.pt and sign_conformer.onnx (+ .data). labels.json is not touched: the class
order comes from the same .npz, so the GRU's labels.json fits this model too.
"""
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from common import cross_language_confusions
from train_gru import prep_raw, augment_raw_batch, evaluate, print_lang_accuracy, export_onnx
from body import prep_raw_body, augment_raw_batch_body

FEATURES = {"hands": (prep_raw, augment_raw_batch),               # 126 hand numbers per frame
            "body": (prep_raw_body, augment_raw_batch_body)}      # + face/body points (body.py), 165 per frame


class FeedForward(nn.Module):
    def __init__(self, d, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mult * d), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(mult * d, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """Pointwise conv + GLU -> depthwise conv over time -> norm -> SiLU -> pointwise conv."""

    def __init__(self, d, kernel=15, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # (B, T, d)
        h = self.norm(x).transpose(1, 2)                    # (B, d, T)
        h = nn.functional.glu(self.pw1(h), dim=1)
        h = self.pw2(nn.functional.silu(self.bn(self.dw(h))))
        return self.drop(h.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d, heads=4, kernel=15, dropout=0.1):
        super().__init__()
        self.ff1, self.ff2 = FeedForward(d, dropout=dropout), FeedForward(d, dropout=dropout)
        self.att_norm = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.att_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d, kernel, dropout)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        h = self.att_norm(x)
        x = x + self.att_drop(self.att(h, h, h, need_weights=False)[0])
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class SignConformer(nn.Module):
    def __init__(self, n_feat, n_classes, d=128, blocks=3, heads=4, kernel=15, dropout=0.2, max_len=60):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, kernel, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d, n_classes))

    def forward(self, x):                                   # x: (B, 60, F)
        h = self.inp(x) + self.pos[:, : x.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.head(torch.cat([h.mean(1), h.max(1).values], dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_prepared.npz")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=5, help="epochs of linear learning-rate warm-up")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=128, help="model width")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--features", choices=list(FEATURES), default="hands",
                    help="hands: hand points only; body: hands + face/body points (needs a body .npz)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(a.data)
    if "X_raw" not in d:
        raise SystemExit("This .npz was made by the old prepare.py - re-run prepare.py to get X_raw.")
    R, y, split, words = d["X_raw"].astype(np.float32), d["y"].astype(np.int64), d["split"], d["words"]
    rng = np.random.default_rng(a.seed)
    mk = lambda x, s, sh: DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y[split == s])),
                                     batch_size=a.batch, shuffle=sh)
    tr = mk(R[split == "train"], "train", True)
    prep, aug_batch = FEATURES[a.features]
    print(f"features: {a.features} ({R.shape[2]} raw numbers per frame)", flush=True)
    tr_clean = mk(prep(R[split == "train"]), "train", False)
    va, te = mk(prep(R[split == "val"]), "val", False), mk(prep(R[split == "test"]), "test", False)
    va_stress = mk(aug_batch(R[split == "val"], np.random.default_rng(1)), "val", False)
    te_stress = mk(aug_batch(R[split == "test"], np.random.default_rng(1)), "test", False)
    print(f"{len(words)} words: {list(words)}")
    print(f"train {len(tr.dataset)}  val {len(va.dataset)}  test {len(te.dataset)}", flush=True)

    n_feat = va.dataset[0][0].shape[1]
    model = SignConformer(n_feat, len(words), d=a.d, blocks=a.blocks)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    warm = max(1, a.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, a.epochs - warm))))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, bad = 0.0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for rb, yb in tr:
            xb = torch.from_numpy(aug_batch(rb.numpy(), rng))
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
            torch.save(model.state_dict(), "conformer_best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    model.load_state_dict(torch.load("conformer_best.pt"))
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
    print("confused across languages (clean test): " + (", ".join(f"{a_} -> {b} x{n}" for a_, b, n in cross) or "none"))
    cm = np.zeros((len(words), len(words)), int)
    np.add.at(cm, (t, p), 1)
    np.fill_diagonal(cm, 0)
    for i, j in zip(*np.unravel_index(np.argsort(cm, axis=None)[::-1][:8], cm.shape)):
        if cm[i, j]:
            print(f"  {words[i]:>15s} -> predicted {words[j]:<15s} x{cm[i, j]}")

    model.eval()
    export_onnx(model, R.shape[1], n_feat, path="sign_conformer.onnx")
    print("saved conformer_best.pt, sign_conformer.onnx")


if __name__ == "__main__":
    main()
