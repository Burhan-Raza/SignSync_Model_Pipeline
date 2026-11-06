"""Classical baseline (no deep learning): summarize each 60-frame sequence into a fixed
vector and train a scikit-learn classifier. The GRU (train_gru.py) should beat it later.

What changed vs. the first version (why live webcam results were wrong):
  * training data is AUGMENTED: mirrored (selfie-flip / other hand), shifted and zoomed in the
    image, faster/slower, cut at the ends, with dropped hand frames - the things that differ
    between the dataset recordings and you signing live in front of your own webcam;
  * a "stress test" re-runs val/test with those same changes, so you can see how the model
    will behave live, not only on clean dataset recordings (which were already ~100%);
  * better features: hand presence per segment (one vs. two hands) and the wrist path
    relative to where the sign started (motion direction, independent of camera framing);
  * the trained model is saved (baseline_model.joblib) so live_baseline.py can use it.

Usage:
    python prepare.py --root "$ROOT" --out data8.npz --eight
    python baseline_sklearn.py data8.npz                 # trains, reports, saves baseline_model.joblib
    python baseline_sklearn.py data8.npz --aug 0         # old behaviour, for comparison
"""
import argparse
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from common import normalize, trim_active, mirror, augment_raw, class_lang, lang_accuracy, cross_language_confusions

SEGMENTS = 6


def featurize_one(xn, segments=SEGMENTS):
    """xn: one normalized sequence (60, 130) -> 1-D feature vector."""
    T = len(xn)
    segs = np.array_split(np.arange(T), segments)
    hands = xn[:, :126].reshape(T, 2, 63)
    pres = (np.abs(hands).sum(axis=2) > 0).astype(np.float32)          # (T, 2)
    wr = xn[:, 126:130].reshape(T, 2, 2).copy()                         # wrist x,y per hand
    # wrist path relative to the first frame where that hand is visible (motion, not framing)
    for h in range(2):
        idx = np.where(pres[:, h] > 0)[0]
        wr[:, h] = (wr[:, h] - wr[idx[0], h]) if len(idx) else 0.0
        wr[pres[:, h] == 0, h] = 0.0
    rel = wr.reshape(T, 4)
    f = [xn[s].mean(axis=0) for s in segs]                              # shape + position per segment
    f += [rel[s].mean(axis=0) for s in segs]                            # movement per segment
    f += [pres[s].mean(axis=0) for s in segs]                           # 1 or 2 hands, when
    f += [xn.std(axis=0), np.abs(np.diff(xn, axis=0)).mean(axis=0)]     # how much / how fast it moves
    return np.concatenate(f).astype(np.float32)


def featurize(Xn):
    return np.stack([featurize_one(x) for x in Xn])


def prep_raw(raw):
    """Same steps live_baseline.py does on webcam input: trim empty ends -> normalize."""
    return normalize(trim_active(raw))


def augmented_set(R, y, n_aug, rng, mirror_p=0.5):
    """Original + mirrored copy + n_aug random augmented copies of every training sequence."""
    feats, labels = [], []
    for r, lab in zip(R, y):
        variants = [r, mirror(r)] if n_aug > 0 else [r]
        for _ in range(n_aug):
            a = augment_raw(r, rng)
            variants.append(mirror(a) if rng.random() < mirror_p else a)
        for v in variants:
            feats.append(featurize_one(prep_raw(v)))
            labels.append(lab)
    return np.stack(feats), np.array(labels)


def stress_set(R, rng):
    """'Like live' version of a val/test set: every sequence randomly augmented, half mirrored."""
    out = []
    for r in R:
        a = augment_raw(r, rng)
        out.append(featurize_one(prep_raw(mirror(a) if rng.random() < 0.5 else a)))
    return np.stack(out)


def build(name):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                n_jobs=-1, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="data_prepared.npz")
    ap.add_argument("--aug", type=int, default=6, help="augmented copies per training sequence (0 = off)")
    ap.add_argument("--model", choices=["extratrees", "logreg"], default="extratrees")
    ap.add_argument("--out", default="baseline_model.joblib")
    ap.add_argument("--train-on-all", action="store_true",
                    help="after reporting, refit on train+val+test (all 3 cameras) for the live model")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    d = np.load(a.data, allow_pickle=False)
    if "X_raw" not in d:
        raise SystemExit("This .npz was made by the old prepare.py - re-run prepare.py to get X_raw.")
    R, y, split, words = d["X_raw"], d["y"], d["split"], d["words"]
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"{len(words)} words: {list(words)}")
    print(f"train {tr.sum()}  val {va.sum()}  test {te.sum()}  | augmentation copies: {a.aug}")

    Xtr, ytr = augmented_set(R[tr], y[tr], a.aug, rng)
    clean = {s: featurize(np.stack([prep_raw(r) for r in R[m]])) for s, m in [("val", va), ("test", te)]}
    stress = {s: stress_set(R[m], np.random.default_rng(1)) for s, m in [("val", va), ("test", te)]}

    m = build(a.model)
    m.fit(Xtr, ytr)
    train_clean = featurize(np.stack([prep_raw(r) for r in R[tr]]))
    print(f"{a.model:10s} train clean acc {accuracy_score(y[tr], m.predict(train_clean)):.3f}   "
          f"(augmented training copies {accuracy_score(ytr, m.predict(Xtr)):.3f})")
    for s, mask in [("val", va), ("test", te)]:
        print(f"{a.model:10s} {s:4s}  clean acc {accuracy_score(y[mask], m.predict(clean[s])):.3f}   "
              f"stress (live-like) acc {accuracy_score(y[mask], m.predict(stress[s])):.3f}")

    present = np.unique(y[te])
    for title, X in [("clean test (webcam source)", clean["test"]), ("STRESS test", stress["test"])]:
        p = m.predict(X)
        print(f"\nPer-word report - {title}:")
        print(classification_report(y[te], p, labels=present, target_names=words[present], zero_division=0))
        cm = confusion_matrix(y[te], p, labels=present)
        w = max(len(x) for x in words[present])
        cw = 9 if any(class_lang(str(x)) for x in words[present]) else 5   # room for the psl_/asl_ prefix
        print("confusion (rows = true, cols = predicted):")
        print(" " * (w + 1) + " ".join(f"{x[:cw]:>{cw}s}" for x in words[present]))
        for i, row in enumerate(cm):
            print(f"{words[present][i]:>{w}s} " + " ".join(f"{v:{cw}d}" for v in row))
        print("accuracy by language: " + "  ".join(f"{k} {v:.3f}" for k, v in lang_accuracy(y[te], p, words).items()))
        if any(class_lang(str(x)) for x in words):
            cross = cross_language_confusions(y[te], p, words)
            print("confused across languages: " + (", ".join(f"{a} -> {b} x{n}" for a, b, n in cross) or "none"))

    if a.train_on_all:
        Xall, yall = augmented_set(R, y, a.aug, rng)
        m = build(a.model).fit(Xall, yall)
        print("\nrefit on all sources (train+val+test) for the live model")
    joblib.dump({"model": m, "words": [str(w) for w in words], "trimmed": bool(d["trimmed"])
                 if "trimmed" in d else True}, a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
