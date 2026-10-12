"""
live_clean.py - live webcam test with the NEW model and the same features.py used in training.
Put it in D:\\ISL\\training\\ and run from D:\\ISL:   python training\\live_clean.py
Keys: q = quit.   The picture is mirrored for display only; the model sees the un-mirrored frame.
"""
import argparse, json, time, collections
from pathlib import Path
import cv2, joblib, numpy as np
import mediapipe as mp
from features import hands_from_mediapipe, build_features

BASE = Path(__file__).resolve().parent.parent          # D:\ISL
ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="model", help="folder (inside D:\\ISL) with the trained model files")
ap.add_argument("--smooth", type=int, default=0, help="average the scores over the last N frames (try 8); 0 = old behaviour")
args = ap.parse_args()
MODEL_DIR = BASE / args.model_dir
TASK_FILE = BASE / "model" / "hand_landmarker.task"
cfg = json.load(open(MODEL_DIR / "config.json"))
labels = json.load(open(MODEL_DIR / "labels.json"))
clf = joblib.load(MODEL_DIR / "isl_clean_mlp.joblib")
THRESH = cfg["threshold"]          # 0.8 - change it in config.json after you read the confidence table
STABLE_FRAMES = 5                  # the same letter must win this many frames in a row

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK_FILE)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam (is another app using it?)")
recent = collections.deque(maxlen=STABLE_FRAMES)
prob_hist = collections.deque(maxlen=max(args.smooth, 1))

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.monotonic() * 1000))
        shown, line2 = "no hand", ""
        if res.hand_landmarks:
            for hand in res.hand_landmarks:
                for p in hand:
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)
            x = build_features(hands_from_mediapipe(res.hand_landmarks), cfg["two_hand"], cfg["rotate"], aspect=w / h, extras=cfg.get("extras", False))
            if x is not None:
                probs = clf.predict_proba(x.reshape(1, -1))[0]
                if args.smooth > 0:
                    prob_hist.append(probs)
                    probs = np.mean(prob_hist, axis=0)
                top = np.argsort(probs)[::-1][:2]
                letter, conf = labels[top[0]], float(probs[top[0]])
                if args.smooth > 0:
                    shown = letter if len(prob_hist) == args.smooth and conf >= THRESH else "Uncertain"
                else:
                    recent.append(letter if conf >= THRESH else None)
                    stable = len(recent) == STABLE_FRAMES and len(set(recent)) == 1 and recent[0] is not None
                    shown = letter if stable else "Uncertain"
                line2 = f"best {letter} {conf:.2f} | next {labels[top[1]]} {probs[top[1]]:.2f}"
        else:
            recent.clear()
            prob_hist.clear()
        view = cv2.flip(frame, 1)                       # mirror for display only
        cv2.putText(view, shown, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        cv2.putText(view, line2, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("ISL live test (q to quit)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
