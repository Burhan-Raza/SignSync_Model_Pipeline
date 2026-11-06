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



def list_sequences(root, sources=SOURCES, min_words_per_source=1, words=None):
    """List sequence folders, optionally scanning only selected words."""
    rows = []
    selected_words = set(words) if words else None

    for src in sources:
        sdir = os.path.join(root, src)
        if not os.path.isdir(sdir):
            continue

        for word in sorted(os.listdir(sdir)):
            # Skip unselected words before scanning their sequences
            if selected_words is not None and word not in selected_words:
                continue

            wdir = os.path.join(sdir, word)
            if not os.path.isdir(wdir) or word in JUNK_FOLDERS:
                continue

            for seq in os.listdir(wdir):
                p = os.path.join(wdir, seq)
                if os.path.isdir(p):
                    n = sum(
                        f.endswith(".npy")
                        for f in os.listdir(p)
                    )
                    rows.append({
                        "source": src,
                        "word": word,
                        "seq_id": seq,
                        "path": p,
                        "n_frames": n,
                    })

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
        scale = np.where(scale > 1e-6, scale, 1.0)[:, None, None]
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
