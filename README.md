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

Predictions are averaged over ten frames (`--smooth 10`) and must remain stable across seven consecutive frames (`STABLE_FRAMES = 7`) before updating, providing smooth, jitter-free output.

## Combined Mode Arbitration Logic

In combined mode (`c`), all three models (Alphabets, Digits, Phrases) run simultaneously on the webcam stream. To prevent cross-model conflicts and misclassifications, the engine applies intelligent multi-stage arbitration:

1. **Structural Hand-Count Filtering**:
   - In ISL, digits (0–9) and specific phrases (`please`, `this`, `mistake`) are strictly single-handed signs. When two hands are detected in the frame, these single-hand candidates are automatically filtered out.
   - Two-handed phrases (`answer`, `book`, `break`, `where`) and two-handed alphabet letters (all except `C`, `I`, `L`, `O`, `U`, `V`) require both hands. When only one hand is visible, two-handed candidates are filtered out.
2. **Letter 'B' vs. Phrase 'break' Resolution**:
   - Letter 'B' is a static two-handed sign where both hands form circles touching each other. When held, the alphabet model produces a stable, confident prediction for 'B'. In this state, letter 'B' takes precedence over phrase 'break'.
   - When performing the dynamic "break" gesture, the hands twist and separate; the alphabet model does not sustain a stable 'B', allowing "break" to be recognized cleanly with high confidence.
3. **Visual Overlap Handling (`please` and `how are you` vs. 'H' / '9')**:
   - The open-hand / chest gestures for `please` and `how are you` visually resemble letter 'H' in the alphabet classifier and digit '9' in the digit classifier. These phrase candidates take precedence over false-positive 'H'/'9' matches.
4. **Alphabet Aspect Ratio Alignment**:
   - Alphabet feature extraction uses `aspect = 1.0` matching `model_aug`'s training configuration, preventing horizontal stretching and eliminating false-positive distortions across letters.

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
test metrics are not included here. This demo classifies static alphabet signs,
digits, and selected phrase gestures; it does not translate continuous signing
or full ISL sentences.

