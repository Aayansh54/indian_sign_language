# ISL Hand Sign Classifier

A Python webcam demo for recognizing Indian Sign Language (ISL) gestures from
MediaPipe hand landmarks. Supports real-time recognition across:
- **Alphabets**: 26 letters (A–Z) from `model_aug`
- **Digits**: 10 numbers (0–9) from `model_digits_v3`
- **Phrases**: 10 common phrases from `model_phrases` (`answer`, `book`, `break`, `how are you`, `i need help`, `mistake`, `please`, `stop`, `this`, `where`)

## Run the webcam demo

From the project root, run:

```powershell
python training\live_clean.py
```

### Controls in Webcam Window:
- `c` — Combined mode (evaluates Alphabets, Digits, and Phrases simultaneously; default)
- `a` — Alphabet mode only (A–Z)
- `d` — Digit mode only (0–9)
- `p` — Phrase mode only (10 phrases)
- `q` — Quit

Predictions are averaged over eight frames (`--smooth 8`) and displayed once the averaged
top-class probability reaches the configured threshold.

## Runtime files

- `training/live_clean.py` — unified webcam capture and multi-model inference loop
- `training/features.py` — landmark preprocessing and feature construction
- `model_aug/` — trained alphabet classifier (`isl_clean_mlp.joblib`, `labels.json`, `config.json`)
- `model_digits_v3/` — trained digit classifier (`isl_digits_mlp.joblib`, `labels.json`, `config.json`)
- `model_phrases/` — trained phrase classifier (`isl_phrases_mlp.joblib`, `labels.json`, `config.json`)
- `model/hand_landmarker.task` — MediaPipe hand landmark model

## Model input

The `model_aug` configuration uses two hands, left-to-right ordering, 21
landmarks per hand, and x/y/z coordinates. Preprocessing applies aspect-ratio
correction, wrist-relative translation, and wrist-to-middle-MCP scale
normalization. Rotation is disabled. Fifteen fingertip distance features are
added per hand, along with two hand-presence flags and three relative-wrist
coordinates, for 161 features total. See `training/features.py` and
`model_aug/config.json` for the implementation and settings.

The camera preview is mirrored for display; inference uses the unmirrored
camera frame.

## Dependencies

Python dependencies used by the demo: OpenCV, MediaPipe, NumPy, joblib, and
scikit-learn. This repository does not pin package versions.

## Dataset, metrics, and limitations

The dataset is not included. Dataset origin and redistribution terms should be
documented by the dataset owner before sharing it. Validation and independent
test metrics are not included here. This demo classifies static alphabet signs;
it does not translate continuous signing or full ISL sentences.

