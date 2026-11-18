"""Hands + face/body points, for the ASL model that can see WHERE the hand is on the face/body.

The hand-only pipeline (common.py, 126 numbers per frame) is unchanged. This adds 13 more MediaPipe Holistic
points per frame (both are in the Kaggle asl-signs files and in the live webcam):
    pose: nose, left eye, right eye, left ear, right ear, mouth left, mouth right,
          left shoulder, right shoulder, left elbow, right elbow
    face mesh: forehead (10), chin (152)
Raw frame = 126 hand numbers [left | right] + 13 x 3 body numbers = 165.

Model features per frame (172):
    126  each hand's shape, exactly as common.normalize (wrist-relative, hand-size scaled)
     20  wrist + 4 fingertips (thumb, index, middle, little) of each hand, relative to the nose
     26  the 13 body points relative to the nose
Positions are divided by the signer's own body size (shoulder width for x, nose-to-shoulder height for y),
so they mean "where on the body" and do not depend on camera distance or image shape.
"""
import numpy as np

from common import fill_gaps, hand_presence, mirror, normalize, resample, SEQ_LEN

POSE_IDS = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14]
FACE_IDS = [10, 152]
N_BODY = len(POSE_IDS) + len(FACE_IDS)            # 13
RAW_DIM = 126 + 3 * N_BODY                        # 165
FEAT_DIM = 126 + 20 + 2 * N_BODY                  # 172
NOSE, L_SH, R_SH = 0, 7, 8                        # positions inside the 13 body points
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]   # eyes, ears, mouth corners, shoulders, elbows
HAND_KEYS = [0, 4, 8, 12, 20]                     # wrist, thumb tip, index tip, middle tip, little tip


def _fill(body):
    """Frames where the body was not found (all zero) take the nearest frame that has it."""
    body = body.copy()
    ok = np.abs(body).sum(axis=(1, 2)) > 0
    if not ok.any():
        return body
    idx = np.where(ok)[0]
    nearest = idx[np.abs(np.arange(len(body))[:, None] - idx[None]).argmin(1)]
    return body[nearest]


def _resample_body(body, n):
    T = len(body)
    if T == n:
        return body.copy()
    if T == 1:
        return np.repeat(body, n, axis=0)
    t = np.linspace(0, T - 1, n)
    lo = np.floor(t).astype(int); hi = np.minimum(lo + 1, T - 1); w = (t - lo)[:, None, None]
    return ((1 - w) * body[lo] + w * body[hi]).astype(np.float32)


def split(seq):
    seq = np.asarray(seq, dtype=np.float32)
    return seq[:, :126], seq[:, 126:].reshape(len(seq), N_BODY, 3)


def join(hands, body):
    return np.concatenate([hands, body.reshape(len(body), -1)], axis=1).astype(np.float32)


def parquet_to_body(df):
    """One Kaggle sequence DataFrame -> raw (T, 165). Hands exactly as convert_asl.parquet_to_seq."""
    from convert_asl import parquet_to_seq
    hands = parquet_to_seq(df)
    frames = np.sort(df["frame"].unique())
    pos = {f: i for i, f in enumerate(frames)}
    body = np.full((len(frames), N_BODY, 3), np.nan, np.float32)
    for k, (kind, ids) in enumerate([("pose", POSE_IDS), ("face", FACE_IDS)]):
        offset = 0 if kind == "pose" else len(POSE_IDS)
        part = df[(df["type"] == kind) & (df["landmark_index"].isin(ids))]
        slot = part["landmark_index"].map({j: offset + i for i, j in enumerate(ids)}).to_numpy()
        body[part["frame"].map(pos).to_numpy(), slot] = part[["x", "y", "z"]].to_numpy(np.float32)
    bad = np.isnan(body).any(axis=2)
    body[bad] = 0.0
    return join(hands, body)


def frame_to_body_vector(results):
    """MediaPipe Holistic result of one webcam frame -> (165,), same layout as parquet_to_body."""
    def pts(lms, ids):
        if lms is None:
            return np.zeros((len(ids), 3), np.float32)
        return np.array([[lms.landmark[i].x, lms.landmark[i].y, lms.landmark[i].z] for i in ids], np.float32)

    def hand(lms):
        return pts(lms, range(21)).ravel()

    hands = np.concatenate([hand(results.left_hand_landmarks), hand(results.right_hand_landmarks)])
    body = np.concatenate([pts(results.pose_landmarks, POSE_IDS), pts(results.face_landmarks, FACE_IDS)])
    return np.concatenate([hands, body.ravel()]).astype(np.float32)


def trim_body(seq, pad=2, n=SEQ_LEN):
    """common.trim_active for the hands, and the same frames, resampled the same way, for the body."""
    hands, body = split(seq)
    hands = fill_gaps(hands)
    body = _fill(body)
    pres = hand_presence(hands).any(axis=1)
    idx = np.where(pres)[0]
    a, b = (0, len(hands)) if len(idx) == 0 else (max(idx[0] - pad, 0), min(idx[-1] + pad + 1, len(hands)))
    return join(resample(hands[a:b], n), _resample_body(body[a:b], n))


def mirror_body(seq):
    """Left <-> right, like common.mirror, for hands and body points together."""
    hands, body = split(seq)
    body = body.copy()
    ok = np.abs(body).sum(axis=2) > 0
    body[..., 0] = np.where(ok, 1.0 - body[..., 0], 0.0)
    for i, j in MIRROR_PAIRS:
        body[:, [i, j]] = body[:, [j, i]]
    return join(mirror(hands), body)


def augment_body(seq, rng):
    """common.augment_raw, applied to hands and body points together (same zoom, shift and timing)."""
    hands, body = split(seq)
    T = len(seq)
    h = hands.reshape(T, 2, 21, 3).copy()
    b = body.copy()
    pres = np.abs(h).sum(axis=(2, 3)) > 0
    bok = np.abs(b).sum(axis=2) > 0
    z = rng.uniform(0.8, 1.2)
    shift = rng.uniform(-0.12, 0.12, size=2)
    h[..., :2] = (h[..., :2] - 0.5) * z + 0.5 + shift
    b[..., :2] = (b[..., :2] - 0.5) * z + 0.5 + shift
    h[..., 2] *= z
    b[..., 2] *= z
    h += rng.normal(0, 0.004, size=h.shape).astype(np.float32)
    b += rng.normal(0, 0.004, size=b.shape).astype(np.float32)
    h[~pres] = 0.0
    b[~bok] = 0.0
    hh = h.reshape(T, 126)
    a0 = rng.integers(0, max(1, int(0.12 * T)))
    b0 = T - rng.integers(0, max(1, int(0.12 * T)))
    hh = resample(hh[a0:b0], T)
    b = _resample_body(b[a0:b0], T)
    if rng.random() < 0.3:
        k = rng.integers(0, 2); f0 = rng.integers(0, T - 4)
        hh[f0:f0 + rng.integers(1, 4), k * 63:(k + 1) * 63] = 0
    return join(hh, b)


def normalize_body(seq):
    """Raw (T, 165) -> features (T, 172)."""
    hands, body = split(seq)
    T = len(seq)
    shape = normalize(hands)[:, :126]
    nose = body[:, NOSE, :2]
    sw = np.abs(body[:, L_SH, 0] - body[:, R_SH, 0])
    sh = (body[:, L_SH, 1] + body[:, R_SH, 1]) / 2 - nose[:, 1]
    sx = float(np.median(sw[sw > 1e-3])) if (sw > 1e-3).any() else 0.3
    sy = float(np.median(sh[sh > 1e-3])) if (sh > 1e-3).any() else 0.2
    scale = np.array([sx, sy], np.float32)
    bok = np.abs(body).sum(axis=2) > 0
    rel_body = np.where(bok[..., None], (body[..., :2] - nose[:, None]) / scale, 0.0)
    h = hands.reshape(T, 2, 21, 3)
    pres = np.abs(h).sum(axis=(2, 3)) > 0
    keys = (h[:, :, HAND_KEYS, :2] - nose[:, None, None]) / scale
    keys = np.where(pres[:, :, None, None], keys, 0.0)
    return np.concatenate([shape, keys.reshape(T, 20), rel_body.reshape(T, 2 * N_BODY)], axis=1).astype(np.float32)


def prep_raw_body(raws):
    """Raw (N, 60, 165) -> model input (N, 60, 172): same steps as live prediction."""
    return np.stack([normalize_body(trim_body(r)) for r in raws]).astype(np.float32)


def augment_raw_batch_body(raws, rng, mirror_p=0.5):
    out = []
    for r in raws:
        v = augment_body(r, rng)
        out.append(mirror_body(v) if rng.random() < mirror_p else v)
    return prep_raw_body(out)
