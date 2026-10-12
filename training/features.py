"""
features.py - the ONE cleaning function. Use it in training AND in the live webcam script.
If training and live use different cleaning, predictions will be wrong, so never copy-edit it.

Steps for each hand (this is the "clean-up" step from our plan):
  1. MOVE    subtract the wrist so the wrist is (0,0,0)       -> position no longer matters
  2. SCALE   divide by wrist-to-middle-finger-base distance    -> distance from camera no longer matters
  3. ROTATE  turn the hand so that line points straight up     -> tilt no longer matters (optional)
"""
import numpy as np

WRIST, MIDDLE_MCP, N_POINTS = 0, 9, 21
TIPS = [4, 8, 12, 16, 20]          # thumb, index, middle, ring, pinky fingertips
N_EXTRAS = 15


def sort_hands(hands):
    """Order hands left-to-right (same rule your old inference_utils.py used)."""
    return sorted(hands, key=lambda h: float(h[:, 0].mean()))


def hands_from_flat(flat126):
    """One CSV row (126 numbers: hand slot 0 then slot 1) -> list of (21,3) arrays, only hands that are present."""
    arr = np.asarray(flat126, dtype=np.float32).reshape(2, N_POINTS, 3)
    return sort_hands([h for h in arr if np.any(h != 0)])


def hands_from_mediapipe(hand_landmarks):
    """MediaPipe result.hand_landmarks -> list of (21,3) arrays."""
    return sort_hands([np.array([[p.x, p.y, p.z] for p in h], dtype=np.float32) for h in hand_landmarks])


def mirror(hands):
    """Left/right mirror image of the hands (used only to augment training data)."""
    out = []
    for h in hands:
        m = h.copy()
        m[:, 0] = 1.0 - m[:, 0]
        out.append(m)
    return sort_hands(out)


def clean_hand(h, rotate=True):
    """(21,3) raw points -> (cleaned (21,3), scale). Returns (None, 0) if the hand is degenerate."""
    p = h - h[WRIST]                                   # 1. move
    v = p[MIDDLE_MCP, :2].copy()
    scale = float(np.hypot(v[0], v[1]))
    if scale < 1e-6:
        return None, 0.0
    p = p / scale                                      # 2. scale
    if rotate:                                         # 3. rotate (image y points down, so "up" is -y)
        phi = -np.pi / 2 - np.arctan2(v[1], v[0])
        c, s = np.cos(phi), np.sin(phi)
        p[:, :2] = p[:, :2] @ np.array([[c, s], [-s, c]], dtype=np.float32)
    return p.astype(np.float32), scale


def extra_features(c):
    """15 shape measurements from a CLEANED hand: 5 fingertip-to-wrist distances + 10 fingertip-to-fingertip distances.
    (For O the thumb touches the other fingertips; for C there is a gap. These numbers show that directly.)"""
    tips = c[TIPS]
    to_wrist = np.linalg.norm(tips - c[WRIST], axis=1)
    i, j = np.triu_indices(5, k=1)
    pair = np.linalg.norm(tips[i] - tips[j], axis=1)
    return np.concatenate([to_wrist, pair]).astype(np.float32)


def build_features(hands, two_hand=False, rotate=True, aspect=1.0, extras=False):
    """
    hands : list of (21,3) arrays (use hands_from_flat or hands_from_mediapipe)
    aspect: image width / height. Live: frame_width/frame_height. Training images: see --train-aspect.
            (MediaPipe x and y are each scaled by the image size, so x must be corrected before measuring shapes.)
    Returns a 1-D float32 vector, or None if no usable hand.
      single-hand mode : 63 numbers (+15 if extras) for the largest hand = the one closest to the camera
      two-hand mode    : 2 x (63 [+15]) + 2 "hand present" flags + 3 hand-to-hand offset numbers
    """
    cleaned = []
    for h in hands:
        h = h.copy()
        h[:, 0] *= aspect
        c, scale = clean_hand(h, rotate)
        if c is not None:
            cleaned.append((c, scale, h))
    if not cleaned:
        return None
    if not two_hand:
        c, _, _ = max(cleaned, key=lambda t: t[1])
        v = c.reshape(-1)
        return np.concatenate([v, extra_features(c)]) if extras else v
    width = N_POINTS * 3 + (N_EXTRAS if extras else 0)
    slots = np.zeros((2, width), dtype=np.float32)
    present = np.zeros(2, dtype=np.float32)
    rel = np.zeros(3, dtype=np.float32)
    for i, (c, _, _) in enumerate(cleaned[:2]):
        slots[i] = np.concatenate([c.reshape(-1), extra_features(c)]) if extras else c.reshape(-1)
        present[i] = 1.0
    if len(cleaned) >= 2:
        (_, s0, h0), (_, _, h1) = cleaned[0], cleaned[1]
        rel = ((h1[WRIST] - h0[WRIST]) / s0).astype(np.float32)
    return np.concatenate([slots.reshape(-1), present, rel])


def augment_hands(hands, rng, tilt=0.35, spin=0.2, noise=0.03):
    """
    Make a realistic 'different camera' copy of the hands, for training only (no recording needed):
    a random 3-D tilt (up to ~20 degrees), a small in-plane spin, and a little jitter on every point.
    """
    if not hands:
        return hands
    ax, ay, az = rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), rng.uniform(-spin, spin)
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    centre = np.mean([h[WRIST] for h in hands], axis=0)
    out = []
    for h in hands:
        size = max(float(np.hypot(*(h[MIDDLE_MCP, :2] - h[WRIST, :2]))), 1e-3)
        p = (h - centre) @ R.T + centre
        p = p + rng.normal(0.0, noise * size, p.shape).astype(np.float32)
        out.append(p.astype(np.float32))
    return sort_hands(out)
