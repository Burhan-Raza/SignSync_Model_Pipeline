"""Build the training .npz for the face/body ASL model straight from the Kaggle asl-signs parquet files.

Per frame: 126 hand numbers (same as convert_asl.py) + 13 face/body points (body.py) = 165 numbers.
Same rules as convert_asl.py + prepare_asl.py: a recording is dropped when hands are visible in under 30% of its
frames or it has fewer than 5 frames; sequences are trimmed to the hands and resampled to 60 frames; the split is
by signer (20% of signers test, 10% val, seed 42), so nobody is in two splits.
No intermediate .npy files are written (that step alone takes about an hour for 7,600 recordings).

    python download_asl.py --signs <the 20 signs> --max-per-sign 1000          # parquet files
    python prepare_asl_body.py --out asl20_body.npz                            # this script
    python train_conformer.py --data asl20_body.npz --features body --epochs 80
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from body import normalize_body, parquet_to_body, trim_body

SIGNS = ["bye", "callonphone", "drink", "fine", "happy", "hello", "home", "hungry", "no", "please",
         "sad", "see", "sick", "sleep", "thankyou", "time", "wait", "water", "where", "yes"]


def one(job):
    kaggle, path, pid, sid, sign = job
    full = os.path.join(kaggle, path)
    if not os.path.exists(full):
        return ("missing", path)
    seq = parquet_to_body(pd.read_parquet(full))
    pres = np.abs(seq[:, :126].reshape(len(seq), 2, 63)).sum(2) > 0
    if len(seq) < 5 or pres.any(1).mean() < 0.3:
        return ("drop", path)
    seq = trim_body(seq).astype(np.float64).astype(np.float32)   # as convert_asl.py saves and loads it
    return ("ok", sign, f"{pid}_{sid}", trim_body(seq))           # prepare_asl.py trims again


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", default="asl-signs", help="folder with train.csv and train_landmark_files/")
    ap.add_argument("--signs", nargs="*", default=SIGNS)
    ap.add_argument("--out", default="asl20_body.npz")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(a.kaggle, "train.csv"))
    meta = meta[meta["sign"].isin(a.signs)]
    jobs = [(a.kaggle, p, pid, sid, s) for p, pid, sid, s in meta[["path", "participant_id", "sequence_id", "sign"]].itertuples(index=False, name=None)]
    print(f"{len(jobs)} recordings listed for {len(set(a.signs))} signs", flush=True)
    kept, stats = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(one, jobs, chunksize=40)):
            stats[res[0]] = stats.get(res[0], 0) + 1
            if res[0] == "ok":
                kept.append(res[1:])
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    print(stats)

    # from here on: prepare_asl.py's order and signer split
    kept.sort(key=lambda r: (r[0], r[1]))
    R = np.stack([r[2] for r in kept]).astype(np.float32)
    words_s = [r[0] for r in kept]
    part = np.array([r[1].split("_")[0] for r in kept])
    ids = np.array([f"asl/{w}/{s}" for w, s, _ in kept])
    rng = np.random.default_rng(a.seed)
    ps = rng.permutation(np.unique(part))
    n_te = max(1, round(a.test_frac * len(ps))); n_va = max(1, round(a.val_frac * len(ps)))
    te_p, va_p = set(ps[:n_te]), set(ps[n_te:n_te + n_va])
    split = np.array(["test" if p in te_p else "val" if p in va_p else "train" for p in part])
    words = np.array(sorted(set(words_s)))
    y = np.searchsorted(words, np.array(words_s))
    X = np.stack([normalize_body(r) for r in R]).astype(np.float32)
    np.savez_compressed(a.out, X=X, X_raw=R, y=y, words=words, sources=np.array(["asl:" + p for p in part]),
                        split=split, ids=ids, trimmed=True)
    print(f"{len(R)} recordings, {len(words)} signs, {len(ps)} signers "
          f"(test signers {sorted(map(str, te_p))}, val {sorted(map(str, va_p))})")
    for s in ["train", "val", "test"]:
        m = split == s
        c = np.bincount(y[m], minlength=len(words))
        print(f"  {s:5s}: {m.sum():6d} recordings | per sign min {c.min()} max {c.max()}")
    print(f"X {X.shape}, X_raw {R.shape} -> saved {a.out}")


if __name__ == "__main__":
    main()
