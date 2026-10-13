"""Extract raw MediaPipe hand landmarks from digit class folders."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp

from inference_utils import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=PROJECT_ROOT / "data" / "digits",
    help="root containing one folder per digit class (default: data/digits)",
)
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT / "data" / "digit_landmarks.csv",
    help="new CSV output path (existing files are never overwritten)",
)
parser.add_argument(
    "--model",
    type=Path,
    default=PROJECT_ROOT / "model" / "hand_landmarker.task",
    help="MediaPipe HandLandmarker task file",
)
parser.add_argument(
    "--classes",
    default="123456789",
    help="digit folder names to process; use 0123456789 when class 0 is added",
)
args = parser.parse_args()

data_dir = args.data_dir.resolve()
output_path = args.output.resolve()
model_path = args.model.resolve()
classes = list(args.classes)

if not classes or any(not label.isdigit() for label in classes):
    parser.error("--classes must contain one or more digit characters, such as 123456789")
if len(classes) != len(set(classes)):
    parser.error("--classes must not contain duplicate digits")
if not data_dir.is_dir():
    parser.error(f"digit dataset folder does not exist: {data_dir}")
if not model_path.is_file():
    parser.error(f"MediaPipe task model does not exist: {model_path}")
if output_path == (PROJECT_ROOT / "data" / "landmarks.csv").resolve():
    parser.error("refusing to use the alphabet landmark output path")
if output_path.is_relative_to(data_dir):
    parser.error("output must be outside the digit image dataset folder")
if output_path.exists():
    parser.error(f"output already exists; refusing to overwrite: {output_path}")

header = ["label", "image_path"] + [
    f"h{hand}_p{point}_{axis}"
    for hand in range(2)
    for point in range(21)
    for axis in ("x", "y", "z")
]

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

stats = defaultdict(lambda: {"scanned": 0, "detected": 0, "unreadable": 0, "no_hand": 0})
output_path.parent.mkdir(parents=True, exist_ok=True)

try:
    with output_path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        with HandLandmarker.create_from_options(options) as detector:
            for label in classes:
                class_dir = data_dir / label
                if not class_dir.is_dir():
                    print(f"Digit {label}: folder not found: {class_dir}")
                    continue

                images = sorted(
                    path
                    for path in class_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )

                for image_path in images:
                    stats[label]["scanned"] += 1
                    image = cv2.imread(str(image_path))
                    if image is None:
                        stats[label]["unreadable"] += 1
                        print(f"Unreadable image for digit {label}: {image_path}")
                        continue

                    try:
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = detector.detect(mp_image)
                    except (cv2.error, ValueError) as exc:
                        stats[label]["unreadable"] += 1
                        print(f"Could not process image for digit {label}: {image_path} ({exc})")
                        continue

                    if not result.hand_landmarks:
                        stats[label]["no_hand"] += 1
                        continue

                    row_features = extract_features(result.hand_landmarks)[0]
                    writer.writerow([label, str(image_path), *row_features])
                    stats[label]["detected"] += 1

                counts = stats[label]
                failures = counts["unreadable"] + counts["no_hand"]
                print(
                    f"Digit {label}: scanned={counts['scanned']}, "
                    f"successful detections={counts['detected']}, failures={failures} "
                    f"(unreadable={counts['unreadable']}, no hand={counts['no_hand']})"
                )
except FileExistsError:
    parser.error(f"output already exists; refusing to overwrite: {output_path}")

total_scanned = sum(item["scanned"] for item in stats.values())
total_detected = sum(item["detected"] for item in stats.values())
total_failures = sum(item["unreadable"] + item["no_hand"] for item in stats.values())
print(
    f"\nDigit landmark extraction complete: scanned={total_scanned}, "
    f"successful detections={total_detected}, failures={total_failures}"
)
print(f"CSV saved to: {output_path}")
