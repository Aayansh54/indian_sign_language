"""Live webcam recognizer for ISL Phrases.

Classes:
    book, break, i need help, mistake, please

Controls:
    Q = Quit
"""
import argparse
import collections
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import mediapipe as mp

from features import build_features, hands_from_mediapipe

BASE = Path(__file__).resolve().parent.parent
TASK_FILE = BASE / "model" / "hand_landmarker.task"
DEFAULT_MODEL_DIR = BASE / "model_phrases"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="path to phrase model directory")
parser.add_argument("--smooth", type=int, default=8, help="frame smoothing window length")
parser.add_argument("--threshold", type=float, default=0.75, help="confidence threshold for display")
args = parser.parse_args()

model_dir = args.model_dir.resolve()
config_path = model_dir / "config.json"
labels_path = model_dir / "labels.json"
model_path = model_dir / "isl_phrases_mlp.joblib"

for path in (config_path, labels_path, model_path):
    if not path.is_file():
        raise FileNotFoundError(f"Required model file not found: {path}")

config = json.loads(config_path.read_text(encoding="utf-8"))
labels = [str(l) for l in json.loads(labels_path.read_text(encoding="utf-8"))]
classifier = joblib.load(model_path)

print(f"Loaded phrase model from: {model_dir}")
print(f"Classes: {labels}")
print("Press Q in webcam window to quit.")

STABLE_FRAMES = 5
SMOOTH_FRAMES = max(1, args.smooth)
score_history = collections.deque(maxlen=SMOOTH_FRAMES)
prediction_history = collections.deque(maxlen=STABLE_FRAMES)

if not TASK_FILE.is_file():
    raise FileNotFoundError(f"MediaPipe task model not found: {TASK_FILE}")

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK_FILE)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Could not open webcam. Close other applications using camera.")

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read frame from webcam.")
                break

            height, width = frame.shape[:2]
            aspect = width / height
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.monotonic() * 1000)

            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                timestamp_ms,
            )

            display_label = "No hand"
            details = "Perform a phrase sign"

            if result.hand_landmarks:
                # Draw hand landmarks
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17),
                ]

                for hand in result.hand_landmarks:
                    for start, end in connections:
                        p1, p2 = hand[start], hand[end]
                        cv2.line(
                            frame,
                            (int(p1.x * width), int(p1.y * height)),
                            (int(p2.x * width), int(p2.y * height)),
                            (255, 180, 0),
                            2,
                        )
                    for pt in hand:
                        cv2.circle(
                            frame,
                            (int(pt.x * width), int(pt.y * height)),
                            4,
                            (0, 255, 0),
                            -1,
                        )

                features = build_features(
                    hands_from_mediapipe(result.hand_landmarks),
                    two_hand=config["two_hand"],
                    rotate=config["rotate"],
                    aspect=aspect,
                    extras=config.get("extras", False),
                )

                if features is not None:
                    probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
                    score_history.append(probabilities)

                    averaged = np.mean(np.asarray(score_history), axis=0)
                    best_index = int(np.argmax(averaged))
                    confidence = float(averaged[best_index])
                    prediction = labels[best_index]

                    prediction_history.append(
                        prediction if confidence >= args.threshold else None
                    )

                    if (
                        len(score_history) == SMOOTH_FRAMES
                        and len(prediction_history) == STABLE_FRAMES
                        and all(p == prediction for p in prediction_history)
                        and confidence >= args.threshold
                    ):
                        display_label = prediction.upper()
                        details = f"Confidence: {confidence:.2f} ({prediction})"
                    else:
                        details = f"Uncertain | Best: {prediction} ({confidence:.2f})"
            else:
                score_history.clear()
                prediction_history.clear()

            view = cv2.flip(frame, 1)

            cv2.putText(
                view,
                display_label,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 255),
                3,
            )
            cv2.putText(
                view,
                details,
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                view,
                f"ISL PHRASES ({len(labels)} classes) | Q: quit",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

            cv2.imshow("ISL Phrase Recognizer", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
