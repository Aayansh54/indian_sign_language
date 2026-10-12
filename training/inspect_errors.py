
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import joblib
import shutil

BASE_DIR = Path(r"D:\ISL")
DATA_DIR = BASE_DIR / "data" / "Validation"
HAND_MODEL_PATH = BASE_DIR / "model" / "hand_landmarker.task"
CLASSIFIER_PATH = BASE_DIR / "model" / "isl_alphabet_model.joblib"
OUTPUT_DIR = BASE_DIR / "model" / "misclassified_images"

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Load the existing model; do not retrain.
classifier = joblib.load(CLASSIFIER_PATH)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

# Re-create only the diagnostic output folder.
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

tested = 0
correct = 0
skipped = 0

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    for letter in LETTERS:
        class_dir = DATA_DIR / letter

        if not class_dir.is_dir():
            continue

        image_files = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )[:20]

        for image_path in image_files:
            image = cv2.imread(str(image_path))

            if image is None:
                skipped += 1
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            result = detector.detect(mp_image)

            if not result.hand_landmarks:
                print(f"SKIPPED {letter}/{image_path.name}: no hand detected")
                skipped += 1
                continue

            hands = sorted(
                result.hand_landmarks,
                key=lambda hand: sum(p.x for p in hand) / len(hand)
            )

            features = []
            for i in range(2):
                if i < len(hands):
                    for point in hands[i]:
                        features.extend([point.x, point.y, point.z])
                else:
                    features.extend([0.0] * 63)

            X = np.asarray(features, dtype=np.float32).reshape(1, -1)
            probabilities = classifier.predict_proba(X)[0]
            best_index = int(np.argmax(probabilities))
            prediction = str(classifier.classes_[best_index])
            confidence = float(probabilities[best_index]) * 100

            tested += 1

            if prediction == letter:
                correct += 1
                continue

            print(
                f"WRONG: expected={letter}, predicted={prediction}, "
                f"confidence={confidence:.1f}%, image={image_path.name}"
            )

            # Copy the original image; the dataset remains untouched.
            output_name = (
                f"expected_{letter}_predicted_{prediction}_"
                f"{confidence:.1f}pct_{image_path.name}"
            )
            cv2.imwrite(str(OUTPUT_DIR / output_name), image)

print("\n========== SUMMARY ==========")
print(f"Correct: {correct}/{tested}")
if tested:
    print(f"Accuracy: {correct / tested * 100:.2f}%")
print(f"Skipped: {skipped}")
print(f"Misclassified images saved to: {OUTPUT_DIR}")
