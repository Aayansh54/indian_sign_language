"""Combined webcam recognizer for ISL Alphabets (A-Z), Digits (0-9), and Phrases.

All three models run simultaneously in real-time, allowing seamless recognition
across letters, numbers, and common phrase signs.

Controls:
    C = Combined mode (Alphabets + Digits + Phrases active simultaneously)
    A = Alphabet mode only (A-Z)
    D = Digit mode only (0-9)
    P = Phrase mode only (10 phrases)
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

# Command-line configuration
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model-dir",
    type=Path,
    default=Path("model_aug"),
    help="Path to alphabet model directory (default: model_aug)",
)
parser.add_argument(
    "--digit-model-dir",
    type=Path,
    default=Path("model_digits_v3"),
    help="Path to digit model directory (default: model_digits_v3)",
)
parser.add_argument(
    "--phrase-model-dir",
    type=Path,
    default=Path("model_phrases"),
    help="Path to phrase model directory (default: model_phrases)",
)
parser.add_argument(
    "--smooth",
    type=int,
    default=10,
    help="Number of frames for rolling average probability smoothing (default: 10)",
)
parser.add_argument(
    "--mode",
    choices=["combined", "alphabet", "digit", "phrase"],
    default="combined",
    help="Initial recognizer mode (default: combined)",
)
args = parser.parse_args()

# Resolve relative paths from project root
ALPHABET_MODEL_DIR = (
    args.model_dir if args.model_dir.is_absolute() else BASE / args.model_dir
)
DIGIT_MODEL_DIR = (
    args.digit_model_dir
    if args.digit_model_dir.is_absolute()
    else BASE / args.digit_model_dir
)
PHRASE_MODEL_DIR = (
    args.phrase_model_dir
    if args.phrase_model_dir.is_absolute()
    else BASE / args.phrase_model_dir
)

STABLE_FRAMES = 7
SMOOTH_FRAMES = max(1, args.smooth)

# Structural hand constraints in Indian Sign Language (ISL):
# - Phrases that are strictly single-handed (in dataset and ISL, 0% have 2 hands):
ONE_HAND_PHRASES = {"please", "this", "mistake"}
# - Phrases that are two-handed gestures:
TWO_HAND_PHRASES = {"answer", "book", "break", "where"}
# - In ISL, only these 6 alphabet letters are one-handed:
ONE_HAND_LETTERS = {"C", "I", "L", "O", "U", "V"}


def load_model(model_dir, expected_labels=None, model_filename="isl_clean_mlp.joblib"):
    config_path = model_dir / "config.json"
    labels_path = model_dir / "labels.json"
    model_path = model_dir / model_filename

    for path in (config_path, labels_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = [
        str(label) for label in json.loads(labels_path.read_text(encoding="utf-8"))
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
        raise ValueError(f"Class indices do not match labels.json in {model_dir}")

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
        str(label) for label in json.loads(labels_path.read_text(encoding="utf-8"))
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

    return classifier, config, labels


def load_phrase_model(model_dir):
    config_path = model_dir / "config.json"
    labels_path = model_dir / "labels.json"
    model_path = model_dir / "isl_phrases_mlp.joblib"

    for path in (config_path, labels_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required phrase model file not found: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = [
        str(label) for label in json.loads(labels_path.read_text(encoding="utf-8"))
    ]
    classifier = joblib.load(model_path)

    if classifier.n_features_in_ != config.get("features"):
        raise ValueError(
            f"Phrase feature mismatch: model={classifier.n_features_in_}, "
            f"config={config.get('features')}"
        )

    if list(classifier.classes_) != list(range(len(labels))):
        raise ValueError("Phrase model class indices do not match labels.json")

    return classifier, config, labels


# Load all three trained classifiers.
alphabet_clf, alphabet_cfg, alphabet_labels = load_model(ALPHABET_MODEL_DIR)
digit_clf, digit_cfg, digit_labels = load_digit_model(DIGIT_MODEL_DIR)
phrase_clf, phrase_cfg, phrase_labels = load_phrase_model(PHRASE_MODEL_DIR)

print(f"Loaded alphabet model: {ALPHABET_MODEL_DIR} ({len(alphabet_labels)} classes)")
print(f"Loaded digit model:    {DIGIT_MODEL_DIR} ({len(digit_labels)} classes)")
print(f"Loaded phrase model:   {PHRASE_MODEL_DIR} ({len(phrase_labels)} classes)")
print("\nTri-Model Unified Recognizer ready:")
print(f"  - Alphabets: {''.join(alphabet_labels)}")
print(f"  - Digits:    {''.join(digit_labels)}")
print(f"  - Phrases:   {', '.join(phrase_labels)}")
print("\nControls:")
print("  C = Combined (all three active simultaneously)")
print("  A = Alphabet mode only")
print("  D = Digit mode only")
print("  P = Phrase mode only")
print("  Q = Quit webcam\n")


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
    "phrase": collections.deque(maxlen=SMOOTH_FRAMES),
}

prediction_histories = {
    "alphabet": collections.deque(maxlen=STABLE_FRAMES),
    "digit": collections.deque(maxlen=STABLE_FRAMES),
    "phrase": collections.deque(maxlen=STABLE_FRAMES),
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
    # Phrase model uses a slightly lower effective threshold in combined mode so
    # it can compete against alphabet/digit models that can reach near-100% confidence
    # on the same hand shape (e.g. alphabet 'H' == the 'please' hand shape).
    effective_threshold = threshold * 0.9 if name == "phrase" else threshold

    stable_history = prediction_histories[name]
    stable_history.append(prediction if confidence >= effective_threshold else None)

    accepted = (
        len(history) == SMOOTH_FRAMES
        and len(stable_history) == STABLE_FRAMES
        and all(item == prediction for item in stable_history)
        and confidence >= effective_threshold
    )

    return {
        "name": name,
        "prediction": prediction,
        "confidence": confidence,
        "accepted": accepted,
        "threshold": effective_threshold,
    }


def main():
    if not TASK_FILE.is_file():
        raise FileNotFoundError(f"MediaPipe hand landmark model not found: {TASK_FILE}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK_FILE)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open webcam. Close other apps using the camera.")

    current_mode = args.mode

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a webcam frame.")
                    break

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(time.monotonic() * 1000)

                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    timestamp_ms,
                )

                display_label = "No hand"
                details = "Show a sign clearly"
                badge_color = (180, 180, 180)

                if result.hand_landmarks:
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12),
                        (9, 13), (13, 14), (14, 15), (15, 16),
                        (13, 17), (17, 18), (18, 19), (19, 20),
                        (0, 17),
                    ]

                    # Draw skeleton on unmirrored frame
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

                    # 1. Alphabet Candidate
                    if current_mode in ("combined", "alphabet"):
                        alphabet_candidate = get_candidate(
                            "alphabet",
                            alphabet_clf,
                            alphabet_cfg,
                            alphabet_labels,
                            result.hand_landmarks,
                            float(alphabet_cfg.get("aspect", 1.0)),
                        )
                        if alphabet_candidate is not None:
                            candidates.append(alphabet_candidate)

                    # 2. Digit Candidate
                    if current_mode in ("combined", "digit"):
                        digit_candidate = get_candidate(
                            "digit",
                            digit_clf,
                            digit_cfg,
                            digit_labels,
                            result.hand_landmarks,
                            float(digit_cfg.get("aspect", 1.0)),
                        )
                        if digit_candidate is not None:
                            candidates.append(digit_candidate)

                    # 3. Phrase Candidate
                    if current_mode in ("combined", "phrase"):
                        phrase_candidate = get_candidate(
                            "phrase",
                            phrase_clf,
                            phrase_cfg,
                            phrase_labels,
                            result.hand_landmarks,
                            float(phrase_cfg.get("aspect", width / height)),
                        )
                        if phrase_candidate is not None:
                            candidates.append(phrase_candidate)

                    num_hands = len(result.hand_landmarks)
                    accepted = [item for item in candidates if item["accepted"]]

                    if accepted:
                        if current_mode == "combined":
                            # 1. Structural hand-count filtering:
                            valid = accepted
                            if num_hands >= 2:
                                # Digits and strictly single-handed phrases cannot match when 2 hands are present
                                filtered = [
                                    c for c in valid
                                    if c["name"] != "digit" and c["prediction"] not in ONE_HAND_PHRASES
                                ]
                                if filtered:
                                    valid = filtered
                            elif num_hands == 1:
                                # 2-handed phrases and 2-handed letters cannot match when only 1 hand is present
                                filtered = [
                                    c for c in valid
                                    if not (c["name"] == "phrase" and c["prediction"] in TWO_HAND_PHRASES)
                                    and not (c["name"] == "alphabet" and c["prediction"] not in ONE_HAND_LETTERS)
                                ]
                                if filtered:
                                    valid = filtered

                            # 2. Letter 'B' vs phrase 'break' priority:
                            # In ISL, letter 'B' is a static two-handed sign. When held, the alphabet model
                            # stably outputs 'B'. In contrast, 'break' is an action gesture where individual
                            # frames do not sustain 'B'. If the alphabet model accepts 'B', it takes precedence.
                            b_cand = next(
                                (c for c in valid if c["name"] == "alphabet" and c["prediction"] == "B"),
                                None,
                            )
                            if b_cand is not None:
                                chosen = b_cand
                            else:
                                # 3. Known overlap phrases ('please', 'how are you'):
                                # These signs visually overlap with alphabet 'H' (and digit '9').
                                overlap_phrase = next(
                                    (
                                        c for c in valid
                                        if c["name"] == "phrase" and c["prediction"] in ("please", "how are you")
                                    ),
                                    None,
                                )
                                if overlap_phrase is not None:
                                    chosen = overlap_phrase
                                else:
                                    phrase_cand = next((c for c in valid if c["name"] == "phrase"), None)
                                    digit_cand = next((c for c in valid if c["name"] == "digit"), None)

                                    if phrase_cand is not None:
                                        # If phrase is 'mistake' and digit model is also confident (e.g. digit 5, 6, 8),
                                        # prefer the digit.
                                        if (
                                            phrase_cand["prediction"] == "mistake"
                                            and digit_cand is not None
                                            and digit_cand["confidence"] >= 0.85
                                        ):
                                            chosen = digit_cand
                                        else:
                                            # Otherwise phrase takes precedence (preserves all working phrases)
                                            chosen = phrase_cand
                                    else:
                                        # Alphabet vs Digit
                                        chosen = max(valid, key=lambda item: item["confidence"])
                        else:
                            chosen = max(accepted, key=lambda item: item["confidence"])

                        display_label = chosen["prediction"].upper()
                        source_title = chosen["name"].capitalize()
                        details = f"Confidence: {chosen['confidence']:.2f} | Source: {source_title}"

                        if chosen["name"] == "alphabet":
                            badge_color = (0, 255, 255)   # Yellow/Cyan
                        elif chosen["name"] == "digit":
                            badge_color = (0, 255, 120)   # Green
                        else:
                            badge_color = (0, 190, 255)   # Gold/Amber

                    elif candidates:
                        best = max(candidates, key=lambda item: item["confidence"])
                        best_source = best["name"].capitalize()
                        display_label = "Uncertain"
                        details = f"Uncertain | Best: {best['prediction'].upper()} {best['confidence']:.2f} ({best_source})"
                        badge_color = (150, 150, 150)
                else:
                    clear_histories()

                view = cv2.flip(frame, 1)

                # Draw recognition text
                cv2.putText(
                    view,
                    display_label,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    badge_color,
                    3,
                )

                # Details line
                cv2.putText(
                    view,
                    details,
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                # Header info bar
                header_text = (
                    f"MODE: {current_mode.upper()} "
                    f"| Keys: C:Combined, A:Alpha, D:Digit, P:Phrase, Q:Quit"
                )
                cv2.putText(
                    view,
                    header_text,
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (220, 220, 220),
                    1,
                )

                cv2.imshow("ISL Tri-Model Recognizer (Alphabets + Digits + Phrases)", view)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                elif key in (ord("c"), ord("C")):
                    current_mode = "combined"
                    clear_histories()
                    print("Switched mode to: COMBINED (Alphabets + Digits + Phrases)")
                elif key in (ord("a"), ord("A")):
                    current_mode = "alphabet"
                    clear_histories()
                    print("Switched mode to: ALPHABET (A-Z)")
                elif key in (ord("d"), ord("D")):
                    current_mode = "digit"
                    clear_histories()
                    print("Switched mode to: DIGIT (0-9)")
                elif key in (ord("p"), ord("P")):
                    current_mode = "phrase"
                    clear_histories()
                    print("Switched mode to: PHRASE (10 phrases)")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

