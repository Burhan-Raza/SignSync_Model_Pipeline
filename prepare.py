"""Step 4: pair landmarks with labels, clean, and save train/val/test arrays.

Usage:
    python prepare.py --root "<path>/PakistanSignLanguageDatasetV2/PakistanSignLanguageDatasetV2" \
                      --out data_prepared.npz [--words hello water bring fan you]

Writes one .npz with X (N,60,130 normalized), y (N,), words, sources, split ('train'/'val'/'test'),
plus an audit printout. Pass --words to try the 5-sign pilot first (roadmap Step 2).

Split strategy: the dataset has NO signer IDs, so a true signer-independent split is impossible.
We use:  webcam_data -> test (different camera/session = the hardest, most honest check),
         laptop+mobile -> random stratified 85/15 train/val.
--split random gives a plain stratified 70/15/15 instead (numbers will look higher - be careful
reporting it in the thesis, because near-duplicate recordings can leak across splits).
"""
import argparse
import collections
import os
import numpy as np
from sklearn.model_selection import train_test_split
from common import list_sequences, load_sequence, normalize, hand_presence, trim_active, SEQ_LEN, EIGHT_SIGNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data_prepared.npz")
    ap.add_argument("--words", nargs="*", help="only keep these words (pilot run)")
    ap.add_argument("--eight", action="store_true",
                    help="shortcut for the 8 app signs: " + " ".join(EIGHT_SIGNS))
    ap.add_argument("--no-trim", action="store_true",
                    help="don't cut empty frames at start/end (trimming is on by default and is "
                         "also done at live prediction time, so keep it on)")
    ap.add_argument("--split", choices=["source", "random"], default="source")
    ap.add_argument("--min-hand-frac", type=float, default=0.3,
                    help="drop sequences where a hand is visible in < this fraction of frames")
    ap.add_argument("--min-seqs", type=int, default=10,
                    help="drop words with fewer usable sequences than this (after cleaning)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra-root", nargs="*", default=[],
                    help="your own recordings, folder layout <dir>/<word>/<n>/0..59.npy "
                         "(made with live_baseline.py --collect). Added to the TRAIN split only.")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        ap.error(f"Dataset root does not exist: {a.root}")
    if a.eight:
        a.words = EIGHT_SIGNS

    print("Scanning selected sign folders...", flush=True)
    rows = list_sequences(a.root, words=a.words)
    if a.eight:
        missing = set(EIGHT_SIGNS) - {r["word"] for r in rows}
        if missing:
            print(f"WARNING: these words are not in the dataset (check spelling): {sorted(missing)}")
    print(f"Found {len(rows)} candidate sequences.", flush=True)
    if not rows:
        ap.error("No sequences found. Check --root, word names, and dataset folder structure.")

    # only keep words present in all three sources, so the webcam test set is meaningful
    by_src = collections.defaultdict(set)
    for r in rows:
        by_src[r["source"]].add(r["word"])
    if a.split == "source":
        if len(by_src) < 3:
            ap.error("Source split requires all three sources: laptop_data, mobile_data, and "
                     "webcam_data. Check the dataset folders or use --split random.")
        common_words = sorted(set.intersection(*by_src.values()))
    else:
        common_words = sorted({r["word"] for r in rows})
    dropped_words = sorted({r["word"] for r in rows} - set(common_words))
    rows = [r for r in rows if r["word"] in common_words]
    if not rows or not common_words:
        ap.error("No words are available for the requested split.")
    extra_src = set()
    for er in a.extra_root:
        er = os.path.abspath(er)
        name = "extra:" + os.path.basename(er)
        ex = list_sequences(os.path.dirname(er), sources=[os.path.basename(er)])
        for r in ex:
            r["source"] = name
        ex = [r for r in ex if r["word"] in common_words]
        print(f"extra data {er}: {len(ex)} sequences")
        rows += ex
        extra_src.add(name)

    X, R, y, src, kept, reasons = [], [], [], [], [], collections.Counter()
    total = len(rows)
    for i, r in enumerate(rows, 1):
        if i == 1 or i % 25 == 0 or i == total:
            print(f"Processing sequence {i}/{total} ({i * 100 // total}%)", flush=True)
        if r["n_frames"] != SEQ_LEN and not r["source"].startswith("extra:"):
            reasons["wrong frame count"] += 1
            continue
        seq = load_sequence(r["path"])
        pres = hand_presence(seq).any(axis=1).mean()
        if pres < a.min_hand_frac:
            reasons["hands mostly missing"] += 1
            continue
        if not a.no_trim or r["source"].startswith("extra:"):
            seq = trim_active(seq)
        R.append(seq)                     # raw (60,126): used for augmentation in training
        X.append(normalize(seq))
        y.append(r["word"])
        src.append(r["source"])
        kept.append(f'{r["source"]}/{r["word"]}/{r["seq_id"]}')
    cnt = collections.Counter(y)
    rare = sorted(w for w in common_words if cnt[w] < a.min_seqs)
    keep = [i for i, w in enumerate(y) if w not in rare]
    if not keep:
        ap.error("No usable sequences remain after cleaning. Check the word names and --min-seqs.")
    X = np.stack([X[i] for i in keep]).astype(np.float32); R = np.stack([R[i] for i in keep]); y = [y[i] for i in keep]
    src = [src[i] for i in keep]; kept = [kept[i] for i in keep]
    common_words = [w for w in common_words if w not in rare]
    dropped_words += rare
    words = np.array(common_words)
    y_idx = np.searchsorted(words, np.array(y))
    src = np.array(src)

    print("Creating train/validation/test splits...", flush=True)
    split = np.empty(len(X), dtype=object)
    if a.split == "source":
        if not (src == "webcam_data").any():
            ap.error("No webcam test sequences found.")
        split[src == "webcam_data"] = "test"
        is_extra = np.isin(src, list(extra_src))
        split[is_extra] = "train"
        tr = np.where((src != "webcam_data") & ~is_extra)[0]
        # Each word must have enough samples for stratification.
        counts = collections.Counter(y_idx[tr])
        if any(counts.get(i, 0) < 2 for i in range(len(words))):
            ap.error("At least two training/validation sequences per word are required "
                     "for stratified splitting.")
        tr_i, va_i = train_test_split(tr, test_size=0.15, stratify=y_idx[tr], random_state=a.seed)
        split[tr_i], split[va_i] = "train", "val"
    else:
        idx = np.arange(len(X))
        counts = collections.Counter(y_idx)
        if any(counts.get(i, 0) < 3 for i in range(len(words))):
            ap.error("Random split needs at least three usable sequences per word "
                     "for stratified train/val/test splitting.")
        tr_i, rest = train_test_split(idx, test_size=0.30, stratify=y_idx, random_state=a.seed)
        va_i, te_i = train_test_split(rest, test_size=0.50, stratify=y_idx[rest], random_state=a.seed)
        split[tr_i], split[va_i], split[te_i] = "train", "val", "test"

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"Saving {len(X)} sequences to {a.out}...", flush=True)
    np.savez_compressed(a.out, X=X, X_raw=R, y=y_idx, words=words, sources=src,
                        split=split.astype(str), ids=np.array(kept), trimmed=not a.no_trim)

    print(f"words kept: {len(words)} -> {words.tolist()}" + (f"  (dropped - not in all sources or too few samples: {dropped_words})" if dropped_words else ""))
    print(f"sequences kept: {len(X)}   dropped: {dict(reasons)}")
    print(f"X shape: {X.shape}  (N, frames, features)")
    for s in ["train", "val", "test"]:
        m = split == s
        c = np.bincount(y_idx[m], minlength=len(words))
        print(f"  {s:5s}: {m.sum():5d} seqs | per word min {c.min()} max {c.max()}")
    print(f"saved -> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
