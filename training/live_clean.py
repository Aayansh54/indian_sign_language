"""
live_clean.py - live webcam test with the NEW model and the same features.py used in training.
Put it in D:\\ISL\\training\\ and run from D:\\ISL:   python training\\live_clean.py
Keys: a = alphabet mode, d = digit mode, q = quit. The picture is mirrored for
display only; both models see the un-mirrored frame.
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
STABLE_FRAMES = 5                  # the same letter must win this many frames in a row

# The digit model is independent of --model-dir and is loaded only after D is
# pressed. Keep it cached after the first successful load.
DIGIT_MODEL_DIR = BASE / "model_digits"
digit_state = None


def load_digit_model():
    global digit_state
    if digit_state is not None:
        return digit_state

    digit_cfg = json.loads((DIGIT_MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    digit_labels = json.loads((DIGIT_MODEL_DIR / "labels.json").read_text(encoding="utf-8"))
    digit_clf = joblib.load(DIGIT_MODEL_DIR / "isl_digits_mlp.joblib")
    if digit_cfg.get("features") != 161:
        raise ValueError(f"Expected 161 digit features, got {digit_cfg.get('features')!r}")
    if digit_cfg.get("two_hand") is not True or digit_cfg.get("rotate") is not False or digit_cfg.get("extras") is not True:
        raise ValueError("Digit model preprocessing settings do not match this live feature pipeline")
    if "aspect" not in digit_cfg:
        raise ValueError("Digit config is missing its training aspect ratio")
    digit_labels = [str(label) for label in digit_labels]
    if digit_labels != [str(n) for n in range(1, 10)]:
        raise ValueError(f"Unexpected digit label order: {digit_labels!r}")
    if getattr(digit_clf, "n_features_in_", None) != digit_cfg["features"]:
        raise ValueError("Digit model input size does not match digit config")
    if list(getattr(digit_clf, "classes_", [])) != list(range(len(digit_labels))):
        raise ValueError("Digit model class indices do not match labels.json order")
    digit_state = (digit_clf, digit_cfg, digit_labels)
    print(f"Loaded digit model from {DIGIT_MODEL_DIR}")
    return digit_state

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK_FILE)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam (is another app using it?)")
histories = {
    "alphabet": (collections.deque(maxlen=STABLE_FRAMES), collections.deque(maxlen=max(args.smooth, 1))),
    "digit": (collections.deque(maxlen=STABLE_FRAMES), collections.deque(maxlen=max(args.smooth, 1))),
}
mode = "alphabet"

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.monotonic() * 1000))
        shown, line2 = "no hand", ""
        recent, prob_hist = histories[mode]
        if res.hand_landmarks:
            for hand in res.hand_landmarks:
                for p in hand:
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)
            if mode == "alphabet":
                active_clf, active_cfg, active_labels = clf, cfg, labels
                aspect = w / h  # preserve the existing alphabet inference behavior
            else:
                active_clf, active_cfg, active_labels = digit_state
                aspect = float(active_cfg["aspect"])
            x = build_features(
                hands_from_mediapipe(res.hand_landmarks), active_cfg["two_hand"], active_cfg["rotate"],
                aspect=aspect, extras=active_cfg.get("extras", False)
            )
            if x is not None:
                if len(x) != active_cfg["features"]:
                    raise ValueError(f"{mode} feature length {len(x)} does not match model config {active_cfg['features']}")
                probs = active_clf.predict_proba(x.reshape(1, -1))[0]
                if args.smooth > 0:
                    prob_hist.append(probs)
                    probs = np.mean(prob_hist, axis=0)
                top = np.argsort(probs)[::-1][:2]
                prediction, conf = active_labels[top[0]], float(probs[top[0]])
                if args.smooth > 0:
                    shown = prediction if len(prob_hist) == args.smooth and conf >= active_cfg["threshold"] else "Uncertain"
                else:
                    recent.append(prediction if conf >= active_cfg["threshold"] else None)
                    stable = len(recent) == STABLE_FRAMES and len(set(recent)) == 1 and recent[0] is not None
                    shown = prediction if stable else "Uncertain"
                line2 = f"{mode} | best {prediction} {conf:.2f} | next {active_labels[top[1]]} {probs[top[1]]:.2f}"
            else:
                recent.clear()
                prob_hist.clear()
                shown, line2 = "Uncertain", f"{mode} | show hand clearly"
        else:
            for recent, prob_hist in histories.values():
                recent.clear()
                prob_hist.clear()
        view = cv2.flip(frame, 1)                       # mirror for display only
        cv2.putText(view, shown, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        cv2.putText(view, line2, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("ISL live test | A: alphabet | D: digits 1-9 | Q: quit", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("a"), ord("A")):
            mode = "alphabet"
            for recent, prob_hist in histories.values():
                recent.clear()
                prob_hist.clear()
        elif key in (ord("d"), ord("D")):
            try:
                digit_state = load_digit_model()
                mode = "digit"
                for recent, prob_hist in histories.values():
                    recent.clear()
                    prob_hist.clear()
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Could not enable digit mode: {exc}")
cap.release()
cv2.destroyAllWindows()
