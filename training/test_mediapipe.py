
from pathlib import Path
import cv2
import mediapipe as mp

# Dataset sample
image_path = Path(r"D:\ISL\data\Training\A\0.jpg")
task_path = Path(r"D:\ISL\model\hand_landmarker.task")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

if not task_path.exists():
    raise FileNotFoundError(f"Model file not found: {task_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image_rgb
)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(task_path)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2
)

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    result = detector.detect(mp_image)

print("Image:", image_path)
print("Image dimensions:", image.shape[1], "x", image.shape[0])
print("Hands detected:", len(result.hand_landmarks))

# Save a preview with landmarks
preview = image.copy()
height, width = preview.shape[:2]

for hand_landmarks in result.hand_landmarks:
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(preview, (x, y), 4, (0, 255, 0), -1)

output_path = Path(r"D:\ISL\model\mediapipe_test.jpg")
cv2.imwrite(str(output_path), preview)
print("Preview saved to:", output_path)
