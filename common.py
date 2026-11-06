"""Shared helpers for the SignSync PSL pipeline.

Dataset layout (PakistanSignLanguageDatasetV2):
    <root>/<source>/<word>/<sequence_id>/<frame_idx>.npy
    source  = laptop_data | mobile_data | webcam_data
    frame   = 126 floats = 2 hands x 21 MediaPipe hand landmarks x (x, y, z)
              slot 0 = values[0:63], slot 1 = values[63:126]; a missing hand is all zeros.
    sequence = 60 frames (a few broken ones are shorter and get dropped).
"""
import os
import numpy as np

SOURCES = ["laptop_data", "mobile_data", "webcam_data"]
SEQ_LEN = 60
N_FEAT = 126
JUNK_FOLDERS = {"s", "test_word", "test2"}   # test/garbage folders found in the archive
MIN_HAND_SCALE = 0.02                        # smallest wrist->middle-MCP distance normalize() divides by


def list_sequences(root, sources=SOURCES, min_words_per_source=1, words=None):
    """Return a list of dicts: {source, word, seq_id, path, n_frames}.
    Pass words to scan only those word folders (much faster than scanning all 71)."""
    rows = []
    selected_words = set(words) if words else None
    for src in sources:
        sdir = os.path.join(root, src)
        if not os.path.isdir(sdir):
            continue
        for word in sorted(os.listdir(sdir)):
            wdir = os.path.join(sdir, word)
            if not os.path.isdir(wdir) or word in JUNK_FOLDERS:
                continue
            # Skip unselected words before scanning their sequences
            if selected_words is not None and word not in selected_words:
                continue
            for seq in os.listdir(wdir):
                p = os.path.join(wdir, seq)
                if os.path.isdir(p):
                    n = sum(f.endswith(".npy") for f in os.listdir(p))
                    rows.append(dict(source=src, word=word, seq_id=seq, path=p, n_frames=n))
    return rows


def load_sequence(path):
    """Load one sequence folder as (60, 126). Frames are sorted NUMERICALLY (0,1,2,...,10),
    not alphabetically (0,1,10,11,...) - getting this wrong silently scrambles time order."""
    files = sorted((f for f in os.listdir(path) if f.endswith(".npy")), key=lambda f: int(f[:-4]))
    return np.stack([np.load(os.path.join(path, f)) for f in files]).astype(np.float32)


def normalize(seq):
    """Make landmarks position/scale invariant, per hand, per frame.
    Each hand: subtract wrist (landmark 0), divide by wrist->middle-MCP (landmark 9) distance.
    Missing hands stay zero. Input/output shape (T, 126)."""
    out = seq.copy().reshape(len(seq), 2, 21, 3)
    for h in range(2):
        hand = out[:, h]                                  # (T, 21, 3)
        present = np.abs(hand).sum(axis=(1, 2)) > 0
        wrist = hand[:, :1, :]
        scale = np.linalg.norm(hand[:, 9, :2] - hand[:, 0, :2], axis=1)
        # A hand pointing at the camera looks almost zero-length in 2D; dividing by that gave values up to ~500.
        scale = np.maximum(scale, MIN_HAND_SCALE)[:, None, None]
        hand = (hand - wrist) / scale
        hand[~present] = 0.0
        out[:, h] = hand
    # also keep where the hand is in the image (wrist x,y) - location matters in sign language
    wrists = seq.reshape(len(seq), 2, 21, 3)[:, :, 0, :2].reshape(len(seq), 4)
    return np.concatenate([out.reshape(len(seq), 126), wrists], axis=1)   # (T, 130)


def hand_presence(seq):
    """(T, 2) boolean: is hand slot 0 / slot 1 detected in each frame."""
    s = seq.reshape(len(seq), 2, 63)
    return np.abs(s).sum(axis=2) > 0


# ---------------------------------------------------------------------------
# Robustness helpers (added for live use). Used the SAME way in training and in
# live prediction, so the model sees the same kind of input in both places.
# ---------------------------------------------------------------------------

EIGHT_SIGNS = ["hello", "assalam-o-alaikum", "water", "welldone", "goodmorning",
               "have_a_good_day", "excuseme", "goodbye"]


def fill_gaps(seq, max_gap=8):
    """MediaPipe often loses a hand for a few frames. Fill short gaps (<= max_gap frames)
    of a hand slot by linear interpolation between the frames around them. Shape (T, 126)."""
    seq = np.asarray(seq, dtype=np.float32).copy()
    T = len(seq)
    for h in range(2):
        part = seq[:, h * 63:(h + 1) * 63]
        pres = np.abs(part).sum(axis=1) > 0
        idx = np.where(pres)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= max_gap + 1:
                w = np.linspace(0, 1, b - a + 1)[1:-1, None]
                part[a + 1:b] = (1 - w) * part[a] + w * part[b]
        seq[:, h * 63:(h + 1) * 63] = part
    return seq


def resample(seq, n=SEQ_LEN):
    """Linearly resample a (T, F) sequence to n frames (handles signing faster/slower)."""
    seq = np.asarray(seq, dtype=np.float32)
    T = len(seq)
    if T == n:
        return seq.copy()
    if T == 1:
        return np.repeat(seq, n, axis=0)
    t = np.linspace(0, T - 1, n)
    lo = np.floor(t).astype(int); hi = np.minimum(lo + 1, T - 1); w = (t - lo)[:, None]
    out = (1 - w) * seq[lo] + w * seq[hi]
    # don't blend a real hand with an all-zero "missing" frame
    for h in range(2):
        sl = slice(h * 63, (h + 1) * 63)
        miss_lo = np.abs(seq[lo, sl]).sum(1) == 0
        miss_hi = np.abs(seq[hi, sl]).sum(1) == 0
        out[miss_lo & ~miss_hi, sl] = seq[hi[miss_lo & ~miss_hi], sl]
        out[miss_hi & ~miss_lo, sl] = seq[lo[miss_hi & ~miss_lo], sl]
        out[miss_lo & miss_hi, sl] = 0
    return out.astype(np.float32)


def trim_active(seq, pad=2, n=SEQ_LEN):
    """Cut away leading/trailing frames with no hand, then resample to n frames.
    Live windows rarely line up with the start/end of a sign; this makes the training
    data and the live input start and end at the same place: when the hands appear/disappear."""
    seq = fill_gaps(seq)
    pres = hand_presence(seq).any(axis=1)
    idx = np.where(pres)[0]
    if len(idx) == 0:
        return resample(seq, n)
    a, b = max(idx[0] - pad, 0), min(idx[-1] + pad + 1, len(seq))
    return resample(seq[a:b], n)


def mirror(seq):
    """Mirror a raw (T, 126) sequence left<->right: x -> 1-x and swap the two hand slots.
    This is exactly what happens if a webcam image is flipped (selfie view), or if the
    signer uses the other hand. Training on both versions makes the model indifferent to it."""
    s = np.asarray(seq, dtype=np.float32).reshape(len(seq), 2, 21, 3).copy()
    pres = np.abs(s).sum(axis=(2, 3)) > 0
    s[..., 0] = np.where(pres[:, :, None], 1.0 - s[..., 0], 0.0)
    return s[:, ::-1].reshape(len(seq), 126).copy()


def augment_raw(seq, rng):
    """Random but sign-preserving changes to a raw (60, 126) sequence:
    shift in the image, zoom, speed change / partial cut at the ends, landmark jitter."""
    T = len(seq)
    s = seq.reshape(T, 2, 21, 3).copy()
    pres = np.abs(s).sum(axis=(2, 3)) > 0                      # (T, 2)
    # zoom around the image centre + shift (different distance to camera / framing)
    z = rng.uniform(0.8, 1.2)
    shift = rng.uniform(-0.12, 0.12, size=2)
    s[..., :2] = (s[..., :2] - 0.5) * z + 0.5 + shift
    s[..., 2] *= z
    s += rng.normal(0, 0.004, size=s.shape).astype(np.float32)   # tracking jitter
    s[~pres] = 0.0
    s = s.reshape(T, 126)
    # speed change / start-end cut: take a random sub-window and stretch it back to T
    a = rng.integers(0, max(1, int(0.12 * T)))
    b = T - rng.integers(0, max(1, int(0.12 * T)))
    s = resample(s[a:b], T)
    # randomly drop a few frames of a hand (MediaPipe misses)
    if rng.random() < 0.3:
        h = rng.integers(0, 2); f0 = rng.integers(0, T - 4)
        s[f0:f0 + rng.integers(1, 4), h * 63:(h + 1) * 63] = 0
    return s.astype(np.float32)


# ---------------------------------------------------------------------------
# One model, two sign languages: classes are named psl_<word> / asl_<word>.
# ---------------------------------------------------------------------------

LANGS = ("psl", "asl")


def class_lang(name):
    """'psl_hello' -> 'psl', 'asl_yes' -> 'asl', 'hello' -> None."""
    return name[:3] if name[:4] in ("psl_", "asl_") else None


def strip_lang(name):
    """Word to show the user: 'psl_hello' -> 'hello'."""
    return name[4:] if class_lang(name) else name


def restrict_lang(probs, names, lang="both"):
    """lang psl/asl: only that language's classes can win (others get 0, the rest renormalized)."""
    p = np.asarray(probs, dtype=np.float64).copy()
    if lang == "both":
        return p
    keep = np.array([class_lang(n) == lang for n in names])
    if not keep.any():
        raise ValueError(f"--lang {lang}: this model has no {lang}_* classes. Its classes: {list(names)}")
    p[~keep] = 0.0
    total = p.sum()
    if total > 0:
        return p / total
    p[keep] = 1.0 / keep.sum()
    return p


def lang_accuracy(y_true, y_pred, words):
    """Accuracy over all samples and over the samples of each language, e.g.
    {'all': .9, 'psl': .95, 'asl': .85}. Wrong-language predictions count as errors."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"all": float((y_true == y_pred).mean())}
    for lang in LANGS:
        m = np.array([class_lang(str(words[i])) == lang for i in y_true])
        if m.any():
            out[lang] = float((y_true[m] == y_pred[m]).mean())
    return out


def cross_language_confusions(y_true, y_pred, words):
    """[(true word, predicted word, count), ...] where the prediction is in the OTHER language."""
    pairs = {}
    for t, p in zip(y_true, y_pred):
        a, b = str(words[t]), str(words[p])
        if class_lang(a) and class_lang(b) and class_lang(a) != class_lang(b):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    return sorted(((a, b, n) for (a, b), n in pairs.items()), key=lambda r: -r[2])
