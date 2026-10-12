from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np


FEATURE_COUNT = 2 * 21 * 3


class Landmark(Protocol):
    x: float
    y: float
    z: float


HandLandmarks = Sequence[Landmark]


def extract_features(hands: Iterable[HandLandmarks]) -> np.ndarray:
    """Return the model's left-to-right, zero-padded 126-value feature row."""
    detected_hands = list(hands)
    for hand in detected_hands:
        if len(hand) != 21:
            raise ValueError(
                f"Expected 21 landmarks per hand, got {len(hand)}"
            )

    detected_hands.sort(
        key=lambda hand: sum(point.x for point in hand) / len(hand)
    )

    features = np.zeros((2, 21, 3), dtype=np.float32)
    for hand_index, hand in enumerate(detected_hands[:2]):
        for point_index, point in enumerate(hand):
            features[hand_index, point_index] = (
                point.x,
                point.y,
                point.z,
            )

    return features.reshape(1, FEATURE_COUNT)


class PredictionStabilizer:
    """Require one class to remain confident for consecutive frames."""

    def __init__(self, required_frames: int = 5) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be at least 1")
        self.required_frames = required_frames
        self._candidate: str | None = None
        self._consecutive_frames = 0

    def update(self, candidate: str | None) -> str | None:
        if candidate is None:
            self._candidate = None
            self._consecutive_frames = 0
            return None

        if candidate == self._candidate:
            self._consecutive_frames += 1
        else:
            self._candidate = candidate
            self._consecutive_frames = 1

        if self._consecutive_frames >= self.required_frames:
            return candidate
        return None
