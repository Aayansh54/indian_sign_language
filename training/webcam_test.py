
from datetime import datetime
import os
import cv2
import joblib
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from inference_utils import PredictionStabilizer, extract_features


# =========================================================
# PATHS AND SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSIFIER_PATH = os.path.join(
    BASE_DIR, "model", "isl_alphabet_model.joblib"
)

HAND_MODEL_PATH = os.path.join(
    BASE_DIR, "model", "hand_landmarker.task"
)

CONFIDENCE_THRESHOLD = 0.50
SAVE_DIAGNOSTIC_FRAMES = False


# =========================================================
# LOAD CLASSIFIER
# =========================================================

print("Loading ISL alphabet classifier...")

if not os.path.isfile(CLASSIFIER_PATH):
    raise FileNotFoundError(
        f"Classifier not found: {CLASSIFIER_PATH}"
    )

if not os.path.isfile(HAND_MODEL_PATH):
    raise FileNotFoundError(
        f"MediaPipe model not found: {HAND_MODEL_PATH}"
    )

classifier = joblib.load(CLASSIFIER_PATH)

print("Classifier loaded successfully.")


# =========================================================
# INITIALIZE MEDIAPIPE
# =========================================================

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# =========================================================
# DRAW LANDMARKS
# =========================================================

def draw_landmarks(frame, result):
    height, width = frame.shape[:2]

    if not result.hand_landmarks:
        return

    for hand in result.hand_landmarks:
        for point in hand:
            x = int(point.x * width)
            y = int(point.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


# =========================================================
# START WEBCAM
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    landmarker.close()
    raise RuntimeError(
        "Cannot open webcam. Check camera permissions "
        "or try changing VideoCapture(0) to VideoCapture(1)."
    )

print("Webcam started. Press Q to quit.")

previous_timestamp = 0
stabilizer = PredictionStabilizer(required_frames=5)

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read webcam frame.")
            break

        # IMPORTANT:
        # Keep the original frame unflipped for prediction.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # VIDEO mode requires monotonically increasing integer ms.
        timestamp_ms = int(
            cv2.getTickCount()
            * 1000
            / cv2.getTickFrequency()
        )

        timestamp_ms = int(
            max(timestamp_ms, previous_timestamp + 1)
        )

        previous_timestamp = timestamp_ms

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Predict and draw landmarks on the unmirrored camera frame, matching
        # the orientation used for training and validation.
        draw_landmarks(frame, result)

        # ---------------------------------------------
        # PREDICT
        # ---------------------------------------------

        if result.hand_landmarks:
            features = extract_features(result.hand_landmarks)

            probabilities = classifier.predict_proba(
                features
            )[0]

            top_indices = np.argsort(probabilities)[-3:][::-1]
            best_index = int(top_indices[0])

            confidence = float(
                probabilities[best_index]
            )

            predicted_letter = str(
                classifier.classes_[best_index]
            )

            if confidence >= CONFIDENCE_THRESHOLD:
                stable_letter = stabilizer.update(predicted_letter)
                if stable_letter is None:
                    status_text = "Status: STABILIZING"
                    status_color = (0, 255, 255)
                else:
                    status_text = f"Status: STABLE MODEL OUTPUT {stable_letter}"
                    status_color = (0, 255, 0)
            else:
                stable_letter = stabilizer.update(None)
                status_text = "Status: LOW CONFIDENCE"
                status_color = (0, 165, 255)

            candidate_text = "Top 3: " + " | ".join(
                f"{classifier.classes_[index]} "
                f"{probabilities[index] * 100:.1f}%"
                for index in top_indices
            )
            hand_count_text = (
                f"Detected hands: {len(result.hand_landmarks)}"
            )
        else:
            stabilizer.update(None)
            candidate_text = "Top 3: --"
            hand_count_text = "Detected hands: 0"
            status_text = "Status: SHOW YOUR HAND"
            status_color = (255, 255, 255)

        # ---------------------------------------------
        # OVERLAY
        # ---------------------------------------------

        # Mirror the camera image for the user, then draw text so the overlay
        # remains readable and is not mirrored with the preview.
        display_frame = cv2.flip(frame, 1)

        cv2.rectangle(
            display_frame,
            (10, 10),
            (620, 135),
            (35, 35, 35),
            -1
        )

        cv2.putText(
            display_frame,
            candidate_text,
            (25, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            hand_count_text,
            (25, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            status_text,
            (25, 113),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )

        cv2.imshow(
            "ISL Alphabet Interpreter - Press Q to Quit",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("s"), ord("S")) and SAVE_DIAGNOSTIC_FRAMES:
            diagnostic_dir = os.path.join(
                BASE_DIR,
                "model",
                "webcam_diagnostics",
            )
            os.makedirs(diagnostic_dir, exist_ok=True)
            diagnostic_path = os.path.join(
                diagnostic_dir,
                f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg",
            )
            if not cv2.imwrite(diagnostic_path, display_frame):
                raise OSError(
                    f"Could not save diagnostic frame: {diagnostic_path}"
                )
            print(f"Diagnostic frame saved: {diagnostic_path}")

        if key == ord("q"):
            break

finally:
    camera.release()
    landmarker.close()
    cv2.destroyAllWindows()

    print("Webcam closed.")
