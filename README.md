# ISL Hand Sign Classifier

A Python webcam demo for recognizing the Indian Sign Language alphabet from
MediaPipe hand landmarks. The selected `model_aug` export recognizes 26 static
letters (A–Z).

## Run the webcam demo

From the project root, run:

```powershell
python training\live_clean.py --model-dir model_aug --smooth 8
```

Press `q` in the webcam window to quit. The camera requires local access.
Predictions are averaged over eight frames and displayed once the averaged
top-class probability reaches the configured 0.8 threshold.

## Runtime files

- `training/live_clean.py` — webcam capture and inference loop
- `training/features.py` — landmark preprocessing and feature construction
- `model_aug/config.json` — feature and runtime settings
- `model_aug/labels.json` — output class order (A through Z)
- `model_aug/isl_clean_mlp.joblib` — trained scikit-learn classifier
- `model/hand_landmarker.task` — MediaPipe hand landmark model

The live script loads `model/hand_landmarker.task` even when another
`--model-dir` is selected. The JSON model export and parity fixture are useful
for other runtimes and parity checks; this Python live script loads the joblib
model.

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

## License

Add a license appropriate to the code, trained model, dataset, and MediaPipe
task asset before distributing this repository. Their redistribution terms may
differ.
