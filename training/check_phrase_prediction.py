"""Verify the trained phrase model on real sample images."""
import argparse
from pathlib import Path
import cv2
import joblib
import json
import mediapipe as mp
import numpy as np

from features import build_features, hands_from_mediapipe

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_phrases"
DATA_DIR = BASE_DIR / "data" / "phrases"
TASK_FILE = BASE_DIR / "model" / "hand_landmarker.task"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
parser.add_argument("--samples-per-class", type=int, default=10)
args = parser.parse_args()

model_dir = args.model_dir.resolve()
data_dir = args.data_dir.resolve()

config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
classifier = joblib.load(model_dir / "isl_phrases_mlp.joblib")

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK_FILE)),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)

print(f"\nEvaluating phrase model on {args.samples_per_class} images per class...")
print(f"Model: {model_dir}")
print(f"Classes: {labels}\n")

total_tested = 0
total_correct = 0
total_skipped = 0

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    for label in labels:
        class_dir = data_dir / label
        if not class_dir.is_dir():
            print(f"Folder not found for {label}")
            continue

        images = sorted(
            [p for p in class_dir.glob("*.png")],
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem,
        )[: args.samples_per_class]

        correct = 0
        tested = 0
        skipped = 0

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                skipped += 1
                continue

            h, w = image.shape[:2]
            aspect = w / h
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            if not result.hand_landmarks:
                skipped += 1
                continue

            feat = build_features(
                hands_from_mediapipe(result.hand_landmarks),
                two_hand=config["two_hand"],
                rotate=config["rotate"],
                aspect=aspect,
                extras=config.get("extras", False),
            )

            if feat is None:
                skipped += 1
                continue

            prob = classifier.predict_proba(feat.reshape(1, -1))[0]
            pred_idx = int(np.argmax(prob))
            predicted_label = labels[pred_idx]
            confidence = prob[pred_idx]

            tested += 1
            if predicted_label == label:
                correct += 1
            else:
                print(f"  WRONG: Expected '{label}', Got '{predicted_label}' ({confidence:.2f}) on {img_path.name}")

        acc = (correct / tested * 100) if tested > 0 else 0.0
        total_correct += correct
        total_tested += tested
        total_skipped += skipped
        print(f"Class '{label:<12}': {correct}/{tested} correct ({acc:.1f}%), skipped (no hand): {skipped}")

overall_acc = (total_correct / total_tested * 100) if total_tested > 0 else 0.0
print("\n" + "=" * 45)
print(f"Overall Direct Accuracy: {total_correct}/{total_tested} ({overall_acc:.2f}%)")
print(f"Images skipped (no hand detected): {total_skipped}")
print("=" * 45 + "\n")
