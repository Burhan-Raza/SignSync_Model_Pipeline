"""Convert the Kaggle "Google - Isolated Sign Language Recognition" (asl-signs) data into the
SAME format as PakistanSignLanguageDatasetV2, so the same prepare / baseline / GRU code works.

Kaggle layout (input):
    train.csv                          columns: path, participant_id, sequence_id, sign
    train_landmark_files/<participant>/<sequence>.parquet
        one row per landmark per frame: frame, row_id, type, landmark_index, x, y, z
        type = face (468) | left_hand (21) | pose (33) | right_hand (21)  -> 543 per frame
        missing landmarks are NaN, frame count varies per sequence

Output (same as PSL):
    <out>/<word>/<participant>_<sequence>/0.npy ... 59.npy      each (126,) = [left 63 | right 63]
    missing hand = zeros, only the 2 x 21 hand points are kept (face and pose are dropped)

Usage:
    pip install pandas pyarrow
    python convert_asl.py --kaggle asl-signs/ --out asl_data --signs hello water bye thankyou
    python convert_asl.py --kaggle asl-signs/ --out asl_data --max-per-sign 120     # all 250 signs, capped
    python convert_asl.py --kaggle asl-signs/ --out asl_data --list                 # just print the signs
Then:
    python prepare_asl.py --asl asl_data --out asl.npz
"""
import argparse
import collections
import os
import numpy as np
import pandas as pd
from common import trim_active, SEQ_LEN


def parquet_to_seq(df):
    """One Kaggle sequence DataFrame -> raw (T, 126) in PSL format [left hand | right hand]."""
    df = df[df["type"].isin(["left_hand", "right_hand"])]
    frames = np.sort(df["frame"].unique())
    T = len(frames)
    out = np.zeros((T, 2, 21, 3), np.float32)
    fpos = {f: i for i, f in enumerate(frames)}
    slot = df["type"].map({"left_hand": 0, "right_hand": 1}).to_numpy()
    fi = df["frame"].map(fpos).to_numpy()
    li = df["landmark_index"].to_numpy().astype(int)
    out[fi, slot, li] = df[["x", "y", "z"]].to_numpy(np.float32)
    # a hand with any NaN point in a frame = not detected in that frame -> all zeros (PSL convention)
    bad = np.isnan(out).any(axis=(2, 3))
    out[bad] = 0.0
    return out.reshape(T, 126)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", required=True, help="folder with train.csv and train_landmark_files/")
    ap.add_argument("--out", default="asl_data")
    ap.add_argument("--csv", default="train.csv", help="sequence list inside --kaggle (download_asl.py writes train_selected.csv)")
    ap.add_argument("--signs", nargs="*", help="only these ASL signs (default: all)")
    ap.add_argument("--max-per-sign", type=int, default=0,
                    help="keep at most N sequences per sign, spread over participants (0 = all). "
                         "~100-120 matches the PSL size per sign")
    ap.add_argument("--swap-hands", action="store_true",
                    help="swap left/right slots if your check shows they are reversed vs PSL")
    ap.add_argument("--min-hand-frac", type=float, default=0.3)
    ap.add_argument("--list", action="store_true", help="print the signs and their counts, then exit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(a.kaggle, a.csv))
    if a.list:
        c = meta["sign"].value_counts().sort_index()
        print(f"{len(c)} signs, {len(meta)} sequences, {meta['participant_id'].nunique()} participants")
        print(", ".join(f"{s}({n})" for s, n in c.items()))
        return
    if a.signs:
        missing = set(a.signs) - set(meta["sign"])
        if missing:
            print(f"WARNING: not in asl-signs: {sorted(missing)}")
        meta = meta[meta["sign"].isin(a.signs)]
    if a.max_per_sign:
        # round-robin over participants so every signer is represented
        meta = meta.sample(frac=1, random_state=a.seed)
        meta["r"] = meta.groupby(["sign", "participant_id"]).cumcount()
        meta = meta.sort_values(["sign", "r"]).groupby("sign").head(a.max_per_sign)

    stats, hand_use = collections.Counter(), np.zeros(2)
    for i, row in enumerate(meta.itertuples(index=False)):
        seq = parquet_to_seq(pd.read_parquet(os.path.join(a.kaggle, row.path)))
        if a.swap_hands:
            seq = np.concatenate([seq[:, 63:], seq[:, :63]], axis=1)
        pres = np.abs(seq.reshape(len(seq), 2, 63)).sum(2) > 0
        if len(seq) < 5 or pres.any(1).mean() < a.min_hand_frac:
            stats["dropped: hands mostly missing / too short"] += 1
            continue
        hand_use += pres.mean(0)
        seq = trim_active(seq, n=SEQ_LEN)              # cut empty ends, resample to 60 frames
        d = os.path.join(a.out, row.sign, f"{row.participant_id}_{row.sequence_id}")
        os.makedirs(d, exist_ok=True)
        for t, v in enumerate(seq):
            np.save(os.path.join(d, f"{t}.npy"), v.astype(np.float64))
        stats["saved"] += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(meta)}")
    print(dict(stats))
    if stats["saved"]:
        l, r = hand_use / stats["saved"]
        print(f"hand slots: left visible {l:.0%} of frames, right {r:.0%}  "
              f"(PSL: left 19%, right 96% - if ASL is the other way round, re-run with --swap-hands)")


if __name__ == "__main__":
    main()
