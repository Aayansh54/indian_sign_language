
"""Combined webcam recognizer for ISL alphabets (A-Z) and digits (0-9).
Phrase recognition is intentionally kept in a separate pipeline.

Controls:
    Q = Quit
"""

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

# Existing model locations
ALPHABET_MODEL_DIR = BASE / "model"
DIGIT_MODEL_DIR = BASE / "model_digits_v3"

STABLE_FRAMES = 5
SMOOTH_FRAMES = 8


def load_model(model_dir, expected_labels=None, model_filename="isl_clean_mlp.joblib"):
    config_path = model_dir / "config.json"
    labels_path = model_dir / "labels.json"
    model_path = model_dir / model_filename

    for path in (config_path, labels_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = [
        str(label)
        for label in json.loads(labels_path.read_text(encoding="utf-8"))
    ]
    classifier = joblib.load(model_path)

    if expected_labels is not None and labels != expected_labels:
        raise ValueError(f"Unexpected labels in {model_dir}: {labels}")

    if classifier.n_features_in_ != config.get("features"):
        raise ValueError(
            f"Feature count mismatch in {model_dir}: "
            f"model={classifier.n_features_in_}, "
            f"config={config.get('features')}"
        )

    if list(classifier.classes_) != list(range(len(labels))):
        raise ValueError(
            f"Class indices do not match labels.json in {model_dir}"
        )

    return classifier, config, labels


def load_digit_model(model_dir):
    config_path = model_dir / "config.json"
    labels_path = model_dir / "labels.json"
    model_path = model_dir / "isl_digits_mlp.joblib"

    for path in (config_path, labels_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required digit model file not found: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = [
        str(label)
        for label in json.loads(labels_path.read_text(encoding="utf-8"))
    ]
    classifier = joblib.load(model_path)

    if labels != [str(i) for i in range(10)]:
        raise ValueError(f"Expected digit labels 0-9; found {labels}")

    if classifier.n_features_in_ != config.get("features"):
        raise ValueError(
            f"Digit feature mismatch: model={classifier.n_features_in_}, "
            f"config={config.get('features')}"
        )

    if list(classifier.classes_) != list(range(len(labels))):
        raise ValueError("Digit model class indices do not match labels.json")

    if config.get("features") != 161:
        raise ValueError(
            f"Expected the digit model to use 161 features, "
            f"found {config.get('features')}"
        )

    return classifier, config, labels


# Load both existing classifiers.
alphabet_clf, alphabet_cfg, alphabet_labels = load_model(
    ALPHABET_MODEL_DIR
)
digit_clf, digit_cfg, digit_labels = load_digit_model(
    DIGIT_MODEL_DIR
)

print(f"Loaded alphabet model: {ALPHABET_MODEL_DIR}")
print(f"Loaded digit model: {DIGIT_MODEL_DIR}")
print("Combined prediction enabled: A-Z and 0-9.")
print("Press Q to quit.")


def make_features(hand_landmarks, config, aspect):
    features = build_features(
        hands_from_mediapipe(hand_landmarks),
        two_hand=config["two_hand"],
        rotate=config["rotate"],
        aspect=aspect,
        extras=config.get("extras", False),
    )

    if features is not None and len(features) != config["features"]:
        raise ValueError(
            f"Feature length {len(features)} does not match "
            f"model configuration {config['features']}"
        )

    return features


score_histories = {
    "alphabet": collections.deque(maxlen=SMOOTH_FRAMES),
    "digit": collections.deque(maxlen=SMOOTH_FRAMES),
}

prediction_histories = {
    "alphabet": collections.deque(maxlen=STABLE_FRAMES),
    "digit": collections.deque(maxlen=STABLE_FRAMES),
}


def clear_histories():
    for history in score_histories.values():
        history.clear()
    for history in prediction_histories.values():
        history.clear()


def get_candidate(name, classifier, config, labels, hand_landmarks, aspect):
    features = make_features(hand_landmarks, config, aspect)

    if features is None:
        score_histories[name].clear()
        prediction_histories[name].clear()
        return None

    probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
    history = score_histories[name]
    history.append(probabilities.copy())

    averaged = np.mean(np.asarray(history), axis=0)
    best_index = int(np.argmax(averaged))
    confidence = float(averaged[best_index])
    prediction = labels[best_index]
    threshold = float(config.get("threshold", 0.8))

    stable_history = prediction_histories[name]
    stable_history.append(
        prediction if confidence >= threshold else None
    )

    accepted = (
        len(history) == SMOOTH_FRAMES
        and len(stable_history) == STABLE_FRAMES
        and all(item == prediction for item in stable_history)
        and confidence >= threshold
    )

    return {
        "name": name,
        "prediction": prediction,
        "confidence": confidence,
        "accepted": accepted,
        "threshold": threshold,
    }


if not TASK_FILE.is_file():
    raise FileNotFoundError(
        f"MediaPipe hand landmark model not found: {TASK_FILE}"
    )

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(TASK_FILE)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit(
        "Could not open webcam. Close other apps using the camera."
    )

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as detector:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read a webcam frame.")
                break

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.monotonic() * 1000)

            result = detector.detect_for_video(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                ),
                timestamp_ms,
            )

            display_label = "No hand"
            details = "Show a sign clearly"

            if result.hand_landmarks:
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

                    for point in hand:
                        cv2.circle(
                            frame,
                            (int(point.x * width), int(point.y * height)),
                            4,
                            (0, 255, 0),
                            -1,
                        )

                candidates = []

                alphabet_candidate = get_candidate(
                    "alphabet",
                    alphabet_clf,
                    alphabet_cfg,
                    alphabet_labels,
                    result.hand_landmarks,
                    width / height,
                )

                if alphabet_candidate is not None:
                    candidates.append(alphabet_candidate)

                digit_candidate = get_candidate(
                    "digit",
                    digit_clf,
                    digit_cfg,
                    digit_labels,
                    result.hand_landmarks,
                    float(digit_cfg["aspect"]),
                )

                if digit_candidate is not None:
                    candidates.append(digit_candidate)

                accepted = [
                    item for item in candidates if item["accepted"]
                ]

                if accepted:
                    # Both models are evaluated automatically.
                    # This compares their raw confidence scores, which
                    # may not be calibrated equally across models.
                    chosen = max(
                        accepted,
                        key=lambda item: item["confidence"],
                    )

                    display_label = chosen["prediction"]
                    details = (
                        f"Confidence: {chosen['confidence']:.2f} "
                        f"| Source: {chosen['name']}"
                    )

                elif candidates:
                    best = max(
                        candidates,
                        key=lambda item: item["confidence"],
                    )
                    details = (
                        f"Uncertain | Best: {best['prediction']} "
                        f"{best['confidence']:.2f}"
                    )
            else:
                clear_histories()

            view = cv2.flip(frame, 1)

            cv2.putText(
                view,
                display_label,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3,
            )

            cv2.putText(
                view,
                details,
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                view,
                "COMBINED: A-Z + 0-9 | Q: quit",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow("ISL Alphabet + Digit Recognizer", view)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
