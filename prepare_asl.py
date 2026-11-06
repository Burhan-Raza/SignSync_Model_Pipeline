"""Make a training .npz from converted ASL data (convert_asl.py), in exactly the format that
baseline_sklearn.py and train_gru.py already read. Optionally merge it with a PSL .npz.

ASL has participant IDs, so here we CAN do a true signer-independent split:
whole participants go to test (and val) - nobody in the test set was seen in training.

    python prepare_asl.py --asl asl_data --out asl.npz                         # ASL only
    python prepare_asl.py --asl asl_data --psl data8.npz --out psl_asl.npz     # PSL + ASL in one model
        (labels become psl_<word> / asl_<word>, so PSL "hello" and ASL "hello" stay different classes)
"""
import argparse
import os
import numpy as np
from common import load_sequence, normalize, trim_active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asl", required=True, help="output folder of convert_asl.py")
    ap.add_argument("--psl", help="PSL .npz from prepare.py to merge with (optional)")
    ap.add_argument("--out", default="asl.npz")
    ap.add_argument("--test-frac", type=float, default=0.2, help="fraction of PARTICIPANTS held out for test")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    R, words_s, part, ids = [], [], [], []
    for w in sorted(os.listdir(a.asl)):
        wd = os.path.join(a.asl, w)
        if not os.path.isdir(wd):
            continue
        for s in sorted(os.listdir(wd)):
            R.append(trim_active(load_sequence(os.path.join(wd, s))))
            words_s.append(w); part.append(s.split("_")[0]); ids.append(f"asl/{w}/{s}")
    R = np.stack(R); part = np.array(part)

    # signer-independent split by participant
    rng = np.random.default_rng(a.seed)
    ps = rng.permutation(np.unique(part))
    n_te = max(1, round(a.test_frac * len(ps))); n_va = max(1, round(a.val_frac * len(ps)))
    te_p, va_p = set(ps[:n_te]), set(ps[n_te:n_te + n_va])
    split = np.array(["test" if p in te_p else "val" if p in va_p else "train" for p in part])
    src = np.array(["asl:" + p for p in part])
    print(f"ASL: {len(R)} sequences, {len(set(words_s))} signs, {len(ps)} participants "
          f"(test participants: {sorted(map(str, te_p))}, val: {sorted(map(str, va_p))})")

    if a.psl:
        d = np.load(a.psl, allow_pickle=False)
        names = ["asl_" + w for w in words_s] + ["psl_" + str(d["words"][i]) for i in d["y"]]
        R = np.concatenate([R, d["X_raw"]]); split = np.concatenate([split, d["split"]])
        src = np.concatenate([src, d["sources"]]); ids = ids + list(d["ids"])
        print(f"PSL: {len(d['y'])} sequences, {len(d['words'])} signs merged in")
    else:
        names = words_s
    words = np.array(sorted(set(names)))
    y = np.searchsorted(words, np.array(names))
    X = np.stack([normalize(r) for r in R])
    np.savez_compressed(a.out, X=X, X_raw=R.astype(np.float32), y=y, words=words, sources=src,
                        split=split, ids=np.array(ids), trimmed=True)
    for s in ["train", "val", "test"]:
        m = split == s
        c = np.bincount(y[m], minlength=len(words))
        print(f"  {s:5s}: {m.sum():6d} seqs | per word min {c.min()} max {c.max()}")
    print(f"{len(words)} classes, X {X.shape} -> saved {a.out}")


if __name__ == "__main__":
    main()
