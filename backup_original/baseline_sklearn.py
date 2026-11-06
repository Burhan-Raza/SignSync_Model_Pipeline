"""Quick baseline without a deep-learning library: summarize each 60-frame sequence into a fixed
vector (6 time segments x mean, plus overall std) and train a classical classifier.
Useful as the 'baseline' row in your results table; the GRU (train_gru.py) should beat it.

Usage: python baseline_sklearn.py data_prepared.npz
"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


def featurize(X, segments=6):
    T = X.shape[1]
    segs = np.array_split(np.arange(T), segments)
    seg_means = [X[:, s].mean(axis=1) for s in segs]
    vel = np.diff(X, axis=1)
    return np.concatenate(seg_means + [X.std(axis=1), np.abs(vel).mean(axis=1)], axis=1)


def main(path):
    d = np.load(path, allow_pickle=False)
    X, y, split, words = featurize(d["X"]), d["y"], d["split"], d["words"]
    tr, va, te = split == "train", split == "val", split == "test"
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=0),
    }
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        print(f"{name:20s} val acc {accuracy_score(y[va], m.predict(X[va])):.3f}   "
              f"test acc {accuracy_score(y[te], m.predict(X[te])):.3f}")
    best = models["ExtraTrees"]
    present = np.unique(y[te])
    print("\nPer-word report on test (ExtraTrees):")
    print(classification_report(y[te], best.predict(X[te]), labels=present,
                                target_names=words[present], zero_division=0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_prepared.npz")
