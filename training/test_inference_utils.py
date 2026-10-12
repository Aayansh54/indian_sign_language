import unittest
from types import SimpleNamespace

import numpy as np

from training.inference_utils import (
    FEATURE_COUNT,
    PredictionStabilizer,
    extract_features,
)


def make_hand(center_x: float):
    return [
        SimpleNamespace(
            x=center_x,
            y=index / 20,
            z=-index / 20,
        )
        for index in range(21)
    ]


class ExtractFeaturesTests(unittest.TestCase):
    def test_sorts_whole_hands_and_keeps_landmark_order(self):
        right_hand = make_hand(0.8)
        left_hand = make_hand(0.2)

        features = extract_features([right_hand, left_hand])

        self.assertEqual(features.shape, (1, FEATURE_COUNT))
        np.testing.assert_allclose(features[0, :3], [0.2, 0.0, 0.0])
        np.testing.assert_allclose(features[0, 63:66], [0.8, 0.0, 0.0])
        np.testing.assert_allclose(
            features[0, 3:6],
            [0.2, 1 / 20, -1 / 20],
        )

    def test_zero_pads_absent_hand(self):
        features = extract_features([make_hand(0.4)])

        self.assertEqual(features.shape, (1, FEATURE_COUNT))
        np.testing.assert_array_equal(features[0, 63:], np.zeros(63))

    def test_rejects_incomplete_hand_landmarks(self):
        with self.assertRaisesRegex(ValueError, "Expected 21 landmarks"):
            extract_features([make_hand(0.4)[:20]])


class PredictionStabilizerTests(unittest.TestCase):
    def test_requires_consecutive_frames(self):
        stabilizer = PredictionStabilizer()

        self.assertIsNone(stabilizer.update("A"))
        self.assertIsNone(stabilizer.update("A"))
        self.assertIsNone(stabilizer.update("A"))
        self.assertIsNone(stabilizer.update("A"))
        self.assertEqual(stabilizer.update("A"), "A")

    def test_class_change_and_uncertain_frame_reset_stability(self):
        stabilizer = PredictionStabilizer(required_frames=3)

        self.assertIsNone(stabilizer.update("A"))
        self.assertIsNone(stabilizer.update("A"))
        self.assertIsNone(stabilizer.update("B"))
        self.assertIsNone(stabilizer.update("B"))
        self.assertIsNone(stabilizer.update(None))
        self.assertIsNone(stabilizer.update("B"))
        self.assertIsNone(stabilizer.update("B"))
        self.assertEqual(stabilizer.update("B"), "B")


if __name__ == "__main__":
    unittest.main()
