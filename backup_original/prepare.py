
"""Step 4: Scan selected signs, clean landmarks, split data, and save arrays."""

import argparse
import collections
import os

import numpy as np
from sklearn.model_selection import train_test_split

from common import (
    list_sequences,
    load_sequence,
    normalize,
    hand_presence,
    SEQ_LEN,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data_prepared.npz")
    ap.add_argument("--words", nargs="*", help="Only process selected words")
    ap.add_argument(
        "--split",
        choices=["source", "random"],
        default="source",
    )
    ap.add_argument("--min-hand-frac", type=float, default=0.3)
    ap.add_argument("--min-seqs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        ap.error(f"Dataset root does not exist: {a.root}")

    print("Scanning selected sign folders...", flush=True)

    # Requires the updated common.py with the words parameter.
    rows = list_sequences(a.root, words=a.words)

    if a.words:
        selected = set(a.words)
        rows = [r for r in rows if r["word"] in selected]

    print(f"Found {len(rows)} candidate sequences.", flush=True)

    if not rows:
        ap.error(
            "No sequences found. Check --root, word names, "
            "and dataset folder structure."
        )

    # Find words present in all available sources.
    by_src = collections.defaultdict(set)
    for r in rows:
        by_src[r["source"]].add(r["word"])

    if a.split == "source":
        if len(by_src) < 3:
            ap.error(
                "Source split requires all three sources: "
                "laptop_data, mobile_data, and webcam_data. "
                "Check the dataset folders or use --split random."
            )
        common_words = sorted(set.intersection(*by_src.values()))
    else:
        common_words = sorted({r["word"] for r in rows})

    dropped_words = sorted(
        {r["word"] for r in rows} - set(common_words)
    )
    rows = [r for r in rows if r["word"] in common_words]

    if not rows or not common_words:
        ap.error("No words are available for the requested split.")

    X, y, src, kept = [], [], [], []
    reasons = collections.Counter()
    total = len(rows)

    for i, r in enumerate(rows, 1):
        if i == 1 or i % 25 == 0 or i == total:
            print(
                f"Processing sequence {i}/{total} "
                f"({i * 100 // total}%)",
                flush=True,
            )

        if r["n_frames"] != SEQ_LEN:
            reasons["wrong frame count"] += 1
            continue

        seq = load_sequence(r["path"])

        pres = hand_presence(seq).any(axis=1).mean()
        if pres < a.min_hand_frac:
            reasons["hands mostly missing"] += 1
            continue

        X.append(normalize(seq))
        y.append(r["word"])
        src.append(r["source"])
        kept.append(
            f'{r["source"]}/{r["word"]}/{r["seq_id"]}'
        )

    cnt = collections.Counter(y)
    rare = sorted(
        w for w in common_words if cnt[w] < a.min_seqs
    )

    keep = [i for i, word in enumerate(y) if word not in rare]

    if not keep:
        ap.error(
            "No usable sequences remain after cleaning. "
            "Check the word names and --min-seqs."
        )

    X = np.stack([X[i] for i in keep]).astype(np.float32)
    y = [y[i] for i in keep]
    src = np.array([src[i] for i in keep])
    kept = [kept[i] for i in keep]

    words = np.array(
        [w for w in common_words if w not in rare]
    )
    dropped_words = sorted(set(dropped_words) | set(rare))
    y_idx = np.searchsorted(words, np.array(y))

    print("Creating train/validation/test splits...", flush=True)

    split = np.empty(len(X), dtype="<U5")

    if a.split == "source":
        test_idx = np.where(src == "webcam_data")[0]
        train_val_idx = np.where(src != "webcam_data")[0]

        if len(test_idx) == 0:
            ap.error("No webcam test sequences found.")

        if len(train_val_idx) == 0:
            ap.error("No training/validation sequences found.")

        # Each word must have enough samples for stratification.
        counts = collections.Counter(y_idx[train_val_idx])
        if any(counts.get(i, 0) < 2 for i in range(len(words))):
            ap.error(
                "At least two training/validation sequences per word "
                "are required for stratified splitting."
            )

        tr_i, va_i = train_test_split(
            train_val_idx,
            test_size=0.15,
            stratify=y_idx[train_val_idx],
            random_state=a.seed,
        )

        split[tr_i] = "train"
        split[va_i] = "val"
        split[test_idx] = "test"

    else:
        counts = collections.Counter(y_idx)
        if any(counts.get(i, 0) < 3 for i in range(len(words))):
            ap.error(
                "Random split needs at least three usable sequences "
                "per word for stratified train/val/test splitting."
            )

        idx = np.arange(len(X))
        tr_i, rest = train_test_split(
            idx,
            test_size=0.30,
            stratify=y_idx,
            random_state=a.seed,
        )

        va_i, te_i = train_test_split(
            rest,
            test_size=0.50,
            stratify=y_idx[rest],
            random_state=a.seed,
        )

        split[tr_i] = "train"
        split[va_i] = "val"
        split[te_i] = "test"

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving {len(X)} sequences to {a.out}...", flush=True)

    np.savez_compressed(
        a.out,
        X=X,
        y=y_idx,
        words=words,
        sources=src,
        split=split,
        ids=np.array(kept),
    )

    print("\nPreprocessing complete!")
    print(f"Words kept: {len(words)} -> {words.tolist()}")
    if dropped_words:
        print(f"Dropped words: {dropped_words}")
    print(f"Sequences kept: {len(X)}")
    print(f"Sequences dropped: {dict(reasons)}")
    print(f"X shape: {X.shape} (N, frames, features)")

    for s in ["train", "val", "test"]:
        mask = split == s
        counts = np.bincount(y_idx[mask], minlength=len(words))
        if mask.any():
            print(
                f"{s:5s}: {mask.sum():5d} sequences | "
                f"per-word min {counts.min()} max {counts.max()}"
            )
        else:
            print(f"{s:5s}: 0 sequences")

    print(f"Saved successfully: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
