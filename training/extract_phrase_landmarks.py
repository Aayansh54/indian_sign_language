"""Extract raw MediaPipe hand landmarks from phrase class folders."""
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
    default=PROJECT_ROOT / "data" / "phrases",
    help="root containing one folder per phrase class (default: data/phrases)",
)
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT / "data" / "phrase_landmarks.csv",
    help="CSV output path (default: data/phrase_landmarks.csv)",
)
parser.add_argument(
    "--model",
    type=Path,
    default=PROJECT_ROOT / "model" / "hand_landmarker.task",
    help="MediaPipe HandLandmarker task file",
)
parser.add_argument(
    "--classes",
    nargs="*",
    default=None,
    help="phrase folder names to process; if omitted, all subdirectories are discovered",
)
parser.add_argument(
    "--min-detection-confidence",
    type=float,
    default=0.3,
    help="MediaPipe minimum hand detection confidence (default: 0.3)",
)
parser.add_argument(
    "--min-presence-confidence",
    type=float,
    default=0.3,
    help="MediaPipe minimum hand presence confidence (default: 0.3)",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="overwrite existing output CSV if present",
)
args = parser.parse_args()

data_dir = args.data_dir.resolve()
output_path = args.output.resolve()
model_path = args.model.resolve()

if not data_dir.is_dir():
    parser.error(f"phrase dataset folder does not exist: {data_dir}")
if not model_path.is_file():
    parser.error(f"MediaPipe task model does not exist: {model_path}")
if output_path.exists() and not args.overwrite:
    parser.error(f"output already exists; use --overwrite to replace: {output_path}")

if args.classes:
    classes = sorted(args.classes)
else:
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

if not classes:
    parser.error(f"No phrase classes found in {data_dir}")

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
    min_hand_detection_confidence=args.min_detection_confidence,
    min_hand_presence_confidence=args.min_presence_confidence,
)

stats = defaultdict(lambda: {"scanned": 0, "detected": 0, "unreadable": 0, "no_hand": 0})
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)

    with HandLandmarker.create_from_options(options) as detector:
        for label in classes:
            class_dir = data_dir / label
            if not class_dir.is_dir():
                print(f"Phrase '{label}': folder not found: {class_dir}")
                continue

            images = sorted(
                [
                    p
                    for p in class_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                ],
                key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
            )

            for image_path in images:
                stats[label]["scanned"] += 1
                image = cv2.imread(str(image_path))
                if image is None:
                    stats[label]["unreadable"] += 1
                    print(f"Unreadable image for phrase '{label}': {image_path}")
                    continue

                try:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect(mp_image)
                except (cv2.error, ValueError) as exc:
                    stats[label]["unreadable"] += 1
                    print(f"Could not process image for phrase '{label}': {image_path} ({exc})")
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
                f"Phrase '{label}': scanned={counts['scanned']}, "
                f"detected={counts['detected']}, failures={failures} "
                f"(unreadable={counts['unreadable']}, no hand={counts['no_hand']})"
            )

total_scanned = sum(item["scanned"] for item in stats.values())
total_detected = sum(item["detected"] for item in stats.values())
total_failures = sum(item["unreadable"] + item["no_hand"] for item in stats.values())
print(
    f"\nPhrase landmark extraction complete: scanned={total_scanned}, "
    f"detected={total_detected}, failures={total_failures}"
)
print(f"CSV saved to: {output_path}")
