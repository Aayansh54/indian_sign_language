"""Train and evaluate a separate 0-9 digit classifier from landmark CSV data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier

from features import build_features, hands_from_flat, mirror


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = [str(i) for i in range(10)]
DEFAULT_CSV = ROOT / "data" / "digit_landmarks_0_9.csv"
DEFAULT_OUT = ROOT / "model_digits_v2"
HIDDEN_LAYERS = (128, 64)
GROUP_SPAN = 30
ASPECT = 1.0
TWO_HAND = True
ROTATE = False
EXTRAS = True

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="digit landmark CSV")
parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="new, separate output directory")
parser.add_argument("--group-span", type=int, default=GROUP_SPAN, help="numeric filename indices kept in the same split group")
parser.add_argument("--aspect", type=float, default=ASPECT, help="training image width/height ratio")
parser.add_argument("--test-size", type=float, default=0.15)
parser.add_argument("--validation-size", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--evaluate-only", action="store_true", help="evaluate saved model against its saved held-out split")
args = parser.parse_args()

csv_path = args.csv.resolve()
out_dir = args.out.resolve()


def fail(message):
    parser.error(message)


def dataset_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path):
    # Explicit dtype prevents pandas from converting digit labels to integers.
    frame = pd.read_csv(path, dtype={"label": str})
    expected_prefix = ["label", "image_path", "h0_p0_x", "h0_p0_y", "h0_p0_z"]
    if len(frame.columns) != 128 or list(frame.columns[:5]) != expected_prefix:
        fail(f"unexpected CSV schema in {path}; expected label, image_path, and 126 landmark values")
    if frame.empty:
        fail(f"no rows found in {path}")
    invalid_labels = sorted(set(frame["label"]) - set(DEFAULT_LABELS))
    if invalid_labels:
        fail(f"digit CSV contains labels outside 0-9: {invalid_labels}")

    normalized_paths = frame["image_path"].map(lambda value: str(Path(value)).casefold())
    if normalized_paths.duplicated().any():
        duplicates = int(normalized_paths.duplicated().sum())
        fail(f"found {duplicates} duplicate image_path rows; remove or resolve duplicates before training")

    values = frame.iloc[:, 2:].to_numpy(dtype=np.float32)
    hands = [hands_from_flat(row) for row in values]
    stems = [Path(value).stem for value in frame["image_path"]]
    nonnumeric = [stem for stem in stems if not stem.isdigit()]
    if nonnumeric:
        fail(f"cannot form source groups: image filenames must have numeric stems; examples: {nonnumeric[:5]}")
    return frame, values, hands, np.asarray([int(stem) for stem in stems], dtype=np.int64)


def merge_duplicate_blocks(block_ids, values):
    """Join filename blocks that contain an exactly repeated 126-value landmark row."""
    unique_blocks = sorted(set(int(item) for item in block_ids))
    parent = {block: block for block in unique_blocks}

    def find(block):
        while parent[block] != block:
            parent[block] = parent[parent[block]]
            block = parent[block]
        return block

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_block_for_row = {}
    for block, row in zip(block_ids, values):
        signature = row.tobytes()
        previous = first_block_for_row.setdefault(signature, int(block))
        union(int(block), previous)

    return np.asarray([find(int(block)) for block in block_ids], dtype=np.int64)


def grouped_split(labels, groups, test_size, validation_size, seed):
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        fail("test-size and validation-size must be positive and sum to less than 1")

    indices = np.arange(len(labels))
    first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(first.split(indices, labels, groups))

    val_fraction_of_train_val = validation_size / (1.0 - test_size)
    second = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_train_val, random_state=seed + 1)
    train_rel, val_rel = next(
        second.split(train_val_idx, labels[train_val_idx], groups[train_val_idx])
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    class_set = set(range(len(DEFAULT_LABELS)))
    for name, part in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        missing = sorted(class_set - set(labels[part]))
        if missing:
            fail(f"{name} split is missing classes {missing}; adjust the split seed or group span")

    assert not (set(groups[train_idx]) & set(groups[val_idx]))
    assert not (set(groups[train_idx]) & set(groups[test_idx]))
    assert not (set(groups[val_idx]) & set(groups[test_idx]))
    return train_idx, val_idx, test_idx


def make_features(hands):
    features = []
    keep = []
    for index, sample_hands in enumerate(hands):
        vector = build_features(
            sample_hands,
            two_hand=TWO_HAND,
            rotate=ROTATE,
            aspect=args.aspect,
            extras=EXTRAS,
        )
        if vector is not None:
            features.append(vector)
            keep.append(index)
    if not features:
        fail("no usable landmark rows after preprocessing")
    return np.asarray(features, dtype=np.float32), np.asarray(keep, dtype=np.int64)


def print_metrics(name, clf, x, y, labels):
    predicted = clf.predict(x)
    matrix = confusion_matrix(y, predicted, labels=np.arange(len(labels)))
    report = classification_report(
        y,
        predicted,
        labels=np.arange(len(labels)),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    print(f"\n{name} accuracy: {accuracy_score(y, predicted):.4f} ({len(y)} samples)")
    print("Per-class precision / recall:")
    for label in labels:
        print(f"  {label}: precision={report[label]['precision']:.4f}, recall={report[label]['recall']:.4f}")
    print("Confusion matrix (rows=true, columns=predicted; order 0-9):")
    print(matrix)
    return predicted, matrix, report


if not csv_path.is_file():
    fail(f"digit landmark CSV not found: {csv_path}")

if args.group_span < 1:
    fail("group-span must be at least 1")

frame, raw_values, hands, numeric_stems = load_rows(csv_path)
content_hash = dataset_hash(csv_path)
block_ids = numeric_stems // args.group_span
groups = merge_duplicate_blocks(block_ids, raw_values)
all_features, usable_rows = make_features(hands)

if len(usable_rows) != len(frame):
    print(f"Dropped {len(frame) - len(usable_rows)} rows with degenerate landmarks during preprocessing")
frame = frame.iloc[usable_rows].reset_index(drop=True)
raw_values = raw_values[usable_rows]
groups = groups[usable_rows]
all_features = all_features

label_order = DEFAULT_LABELS
label_to_index = {label: index for index, label in enumerate(label_order)}
y = frame["label"].map(label_to_index).to_numpy(dtype=np.int64)

manifest_path = out_dir / "split_manifest.json"
if args.evaluate_only:
    if not manifest_path.is_file() or not (out_dir / "isl_digits_mlp.joblib").is_file():
        fail(f"saved digit model or split manifest not found in {out_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["csv_sha256"] != content_hash:
        fail("digit CSV differs from the CSV used to train this model; refusing an unmatched evaluation")
    if manifest["feature_settings"]["aspect"] != args.aspect or manifest["group_span"] != args.group_span:
        fail("evaluation preprocessing/group settings differ from the saved model manifest")
    train_idx = np.asarray(manifest["train_indices"], dtype=np.int64)
    val_idx = np.asarray(manifest["validation_indices"], dtype=np.int64)
    test_idx = np.asarray(manifest["test_indices"], dtype=np.int64)
    clf = joblib.load(out_dir / "isl_digits_mlp.joblib")
else:
    if out_dir.exists():
        fail(f"output directory already exists; refusing to overwrite: {out_dir}")
    train_idx, val_idx, test_idx = grouped_split(
        y, groups, args.test_size, args.validation_size, args.seed
    )

    train_x = all_features[train_idx]
    train_y = y[train_idx]
    # Match the existing clean pipeline's horizontal-mirror augmentation, on train rows only.
    mirrored = []
    for row_index in train_idx:
        mirrored_hands = mirror(hands[usable_rows[row_index]])
        vector = build_features(
            mirrored_hands,
            two_hand=TWO_HAND,
            rotate=ROTATE,
            aspect=args.aspect,
            extras=EXTRAS,
        )
        if vector is not None:
            mirrored.append(vector)
    if mirrored:
        train_x = np.concatenate([train_x, np.asarray(mirrored, dtype=np.float32)], axis=0)
        train_y = np.concatenate([train_y, y[train_idx]])

    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        max_iter=400,
        early_stopping=True,
        random_state=args.seed,
    )
    print(
        f"Rows: {len(frame)} | features: {all_features.shape[1]} | "
        f"train={len(train_idx)} (+{len(mirrored)} mirrored), "
        f"validation={len(val_idx)}, test={len(test_idx)} | "
        f"groups={len(np.unique(groups))}"
    )
    clf.fit(train_x, train_y)

    if set(clf.classes_) != set(range(len(label_order))):
        fail("trained classifier does not contain all digit classes")
    val_pred, _, val_report = print_metrics(
        "Validation", clf, all_features[val_idx], y[val_idx], label_order
    )

    out_dir.mkdir(parents=True, exist_ok=False)
    feature_settings = {
        "features": int(all_features.shape[1]),
        "two_hand": TWO_HAND,
        "rotate": ROTATE,
        "extras": EXTRAS,
        "aspect": args.aspect,
        "threshold": 0.8,
        "note": "Features are produced by training/features.py build_features(). Threshold is a default, not validation-calibrated.",
    }
    manifest = {
        "csv_sha256": content_hash,
        "group_span": args.group_span,
        "seed": args.seed,
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "feature_settings": feature_settings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(feature_settings, indent=2), encoding="utf-8")
    (out_dir / "labels.json").write_text(json.dumps(label_order, indent=2), encoding="utf-8")
    joblib.dump(clf, out_dir / "isl_digits_mlp.joblib")
    layers = [
        {
            "W": np.round(weights, 6).tolist(),
            "b": np.round(bias, 6).tolist(),
            "activation": "relu" if layer_index < len(clf.coefs_) - 1 else "softmax",
        }
        for layer_index, (weights, bias) in enumerate(zip(clf.coefs_, clf.intercepts_))
    ]
    (out_dir / "model.json").write_text(json.dumps({"layers": layers}), encoding="utf-8")

    first_test_index = int(test_idx[0])
    first_raw_hands = [hand.tolist() for hand in hands[usable_rows[first_test_index]]]
    first_feature = all_features[first_test_index]
    first_probabilities = clf.predict_proba(first_feature.reshape(1, -1))[0]
    prediction_test = {
        "source_image": str(frame.iloc[first_test_index]["image_path"]),
        "true_label": label_order[int(y[first_test_index])],
        "predicted_label": label_order[int(clf.predict(first_feature.reshape(1, -1))[0])],
        "aspect": args.aspect,
        "raw_hands": first_raw_hands,
        "expected_features": first_feature.tolist(),
        "expected_probabilities": first_probabilities.tolist(),
        "labels": label_order,
    }
    (out_dir / "prediction_test.json").write_text(json.dumps(prediction_test), encoding="utf-8")

    prediction_path = out_dir / "test_predictions.csv"
    test_probabilities = clf.predict_proba(all_features[test_idx])
    test_predictions = clf.predict(all_features[test_idx])
    with prediction_path.open("w", newline="", encoding="utf-8") as pred_file:
        writer = csv.writer(pred_file)
        writer.writerow(["image_path", "true_label", "predicted_label", *[f"p_{label}" for label in label_order]])
        for row_index, predicted, probabilities in zip(test_idx, test_predictions, test_probabilities):
            writer.writerow([
                frame.iloc[row_index]["image_path"],
                label_order[int(y[row_index])],
                label_order[int(predicted)],
                *probabilities.tolist(),
            ])

    test_pred, test_matrix, test_report = print_metrics(
        "Held-out test", clf, all_features[test_idx], y[test_idx], label_order
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "validation_accuracy": float(accuracy_score(y[val_idx], val_pred)),
                "test_accuracy": float(accuracy_score(y[test_idx], test_pred)),
                "validation_report": val_report,
                "test_report": test_report,
                "test_confusion_matrix": test_matrix.tolist(),
                "label_order": label_order,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDigit model and evaluation artifacts saved to: {out_dir}")

if args.evaluate_only:
    print_metrics("Held-out test", clf, all_features[test_idx], y[test_idx], label_order)
    print(f"Saved held-out prediction fixture: {out_dir / 'prediction_test.json'}")

