
from pathlib import Path
import cv2
import mediapipe as mp
import joblib

from inference_utils import extract_features

BASE_DIR = Path(r"D:\ISL")
DATA_DIR = BASE_DIR / "data" / "Validation"
HAND_MODEL_PATH = BASE_DIR / "model" / "hand_landmarker.task"
CLASSIFIER_PATH = BASE_DIR / "model" / "isl_alphabet_model.joblib"

SAMPLES_PER_LETTER = 20
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

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

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
overall_correct = 0
overall_tested = 0
overall_skipped = 0

print("\nTesting 20 validation images per letter...\n")

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    for letter in letters:
        class_dir = DATA_DIR / letter

        if not class_dir.is_dir():
            print(f"{letter}: folder not found")
            continue

        image_files = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )[:SAMPLES_PER_LETTER]

        if not image_files:
            print(f"{letter}: no images found")
            continue

        correct = 0
        tested = 0
        skipped = 0

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
                skipped += 1
                continue

            X = extract_features(result.hand_landmarks)
            prediction = classifier.predict(X)[0]
            if prediction != letter:
                print(
                    f"  WRONG: expected {letter}, "
                    f"predicted {prediction}, "
                    f"image: {image_path.name}"
                    )
            tested += 1
            if prediction == letter:
                correct += 1

        accuracy = (correct / tested * 100) if tested else 0.0

        overall_correct += correct
        overall_tested += tested
        overall_skipped += skipped

        print(
            f"{letter}: {correct}/{tested} correct "
            f"({accuracy:.1f}%), skipped: {skipped}"
        )

print("\n========== SUMMARY ==========")

if overall_tested:
    accuracy = overall_correct / overall_tested * 100
    print(f"Overall accuracy: {overall_correct}/{overall_tested} "
          f"({accuracy:.2f}%)")

print(f"Images skipped (no hand/read error): {overall_skipped}")
print("Testing complete. Existing model was not changed.")
