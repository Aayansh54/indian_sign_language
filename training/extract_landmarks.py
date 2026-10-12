
from pathlib import Path
import csv
import cv2
import mediapipe as mp

from inference_utils import extract_features

DATA_DIR = Path(r"D:\ISL\data\Training")
MODEL_PATH = Path(r"D:\ISL\model\hand_landmarker.task")
OUTPUT_PATH = Path(r"D:\ISL\data\landmarks.csv")

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"MediaPipe model missing: {MODEL_PATH}")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

# Store a fixed-size feature vector:
# 2 hands x 21 landmarks x (x, y, z) = 126 values.
HEADER = (
    ["label", "image_path"]
    + [
        f"h{hand}_p{point}_{axis}"
        for hand in range(2)
        for point in range(21)
        for axis in ("x", "y", "z")
    ]
)

processed = 0
skipped = 0

with HandLandmarker.create_from_options(options) as detector, \
     OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:

    writer = csv.writer(csv_file)
    writer.writerow(HEADER)

    for label in CLASSES:
        class_dir = DATA_DIR / label

        if not class_dir.is_dir():
            print(f"Missing class folder: {class_dir}")
            continue

        image_files = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        for image_path in image_files:
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Unreadable image: {image_path}")
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

            features = extract_features(result.hand_landmarks)

            writer.writerow([label, str(image_path), *features[0]])
            processed += 1

        print(f"Finished {label}: {len(image_files)} images scanned")

print(f"\nLandmark extraction complete.")
print(f"Images with detected hands: {processed}")
print(f"Images skipped: {skipped}")
print(f"CSV saved to: {OUTPUT_PATH}")
