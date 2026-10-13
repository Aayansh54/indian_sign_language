"""Train and evaluate a phrase classifier from landmark CSV data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from features import augment_hands, build_features, hands_from_flat, mirror


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "phrase_landmarks.csv"
DEFAULT_OUT = ROOT / "model_phrases"
HIDDEN_LAYERS = (128, 64)
GROUP_SPAN = 8
ASPECT = 4.0 / 3.0  # 640x480 images have 4:3 aspect ratio
TWO_HAND = True
ROTATE = False
EXTRAS = True

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 64], help="MLP hidden layer sizes")
parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="phrase landmark CSV")
parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for saved model artifacts")
parser.add_argument("--group-span", type=int, default=GROUP_SPAN, help="numeric filename index block span for grouping")
parser.add_argument("--aspect", type=float, default=ASPECT, help="image aspect ratio (width/height)")
parser.add_argument("--test-size", type=float, default=0.2, help="fraction of groups held out for test")
parser.add_argument("--validation-size", type=float, default=0.2, help="fraction of groups held out for validation")
parser.add_argument("--augment", type=int, default=2, help="number of tilted/jittered copies per training sample")
parser.add_argument("--no-mirror", action="store_true", help="disable mirror augmentation")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--evaluate-only", action="store_true", help="evaluate saved model against its saved held-out split")
parser.add_argument("--overwrite", action="store_true", help="overwrite existing output directory if it exists")
args = parser.parse_args()

csv_path = args.csv.resolve()
out_dir = args.out.resolve()


def fail(message: str) -> None:
    parser.error(message)


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path):
    frame = pd.read_csv(path, dtype={"label": str})
    expected_prefix = ["label", "image_path", "h0_p0_x", "h0_p0_y", "h0_p0_z"]
    if len(frame.columns) != 128 or list(frame.columns[:5]) != expected_prefix:
        fail(f"unexpected CSV schema in {path}; expected label, image_path, and 126 landmark values")
    if frame.empty:
        fail(f"no rows found in {path}")

    labels = sorted(frame["label"].unique())
    normalized_paths = frame["image_path"].map(lambda val: str(Path(val)).casefold())
    if normalized_paths.duplicated().any():
        dups = int(normalized_paths.duplicated().sum())
        fail(f"found {dups} duplicate image_path rows in CSV")

    values = frame.iloc[:, 2:].to_numpy(dtype=np.float32)
    hands = [hands_from_flat(row) for row in values]

    stems = [Path(val).stem for val in frame["image_path"]]
    numeric_stems = []
    for stem in stems:
        if stem.isdigit():
            numeric_stems.append(int(stem))
        else:
            # fallback for non-digit stems: hash
            numeric_stems.append(abs(hash(stem)) % 1000)

    return frame, values, hands, np.asarray(numeric_stems, dtype=np.int64), labels


def make_features(hands, aspect):
    features = []
    keep = []
    for index, sample_hands in enumerate(hands):
        vector = build_features(
            sample_hands,
            two_hand=TWO_HAND,
            rotate=ROTATE,
            aspect=aspect,
            extras=EXTRAS,
        )
        if vector is not None:
            features.append(vector)
            keep.append(index)
    if not features:
        fail("no usable landmark rows after preprocessing")
    return np.asarray(features, dtype=np.float32), np.asarray(keep, dtype=np.int64)


def stratified_grouped_split(frame, labels, groups, test_size, validation_size, seed):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for lab in labels:
        class_mask = (frame["label"] == lab).to_numpy()
        class_indices = np.where(class_mask)[0]
        class_groups = np.unique(groups[class_indices])
        rng.shuffle(class_groups)

        n_groups = len(class_groups)
        n_test = max(1, int(round(n_groups * test_size))) if n_groups > 2 else 0
        n_val = max(1, int(round(n_groups * validation_size))) if (n_groups - n_test) > 1 else 0

        test_groups = set(class_groups[:n_test])
        val_groups = set(class_groups[n_test : n_test + n_val])
        train_groups = set(class_groups[n_test + n_val :])

        if not train_groups:
            # ensure at least one group in training
            train_groups = {class_groups[-1]}
            if val_groups and len(val_groups) > 1:
                val_groups.remove(class_groups[-1])
            elif test_groups and len(test_groups) > 1:
                test_groups.remove(class_groups[-1])

        for idx in class_indices:
            grp = groups[idx]
            if grp in test_groups:
                test_idx.append(idx)
            elif grp in val_groups:
                val_idx.append(idx)
            else:
                train_idx.append(idx)

    train_idx = np.asarray(sorted(train_idx), dtype=np.int64)
    val_idx = np.asarray(sorted(val_idx), dtype=np.int64)
    test_idx = np.asarray(sorted(test_idx), dtype=np.int64)

    # Verify no group overlap
    train_g = set(groups[train_idx])
    val_g = set(groups[val_idx])
    test_g = set(groups[test_idx])
    assert not (train_g & val_g), "Group leakage between train and val"
    assert not (train_g & test_g), "Group leakage between train and test"
    assert not (val_g & test_g), "Group leakage between val and test"

    return train_idx, val_idx, test_idx


def print_metrics(name, clf, x, y, labels):
    predicted = clf.predict(x)
    probabilities = clf.predict_proba(x)
    conf = probabilities.max(axis=1)

    matrix = confusion_matrix(y, predicted, labels=np.arange(len(labels)))
    report = classification_report(
        y,
        predicted,
        labels=np.arange(len(labels)),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    acc = accuracy_score(y, predicted)

    print(f"\n===== {name}: {len(y)} samples =====")
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nPer-class precision / recall / f1-score:")
    for label in labels:
        p = report[label]["precision"]
        r = report[label]["recall"]
        f1 = report[label]["f1-score"]
        supp = report[label]["support"]
        print(f"  {label:<15}: precision={p:.4f}, recall={r:.4f}, f1={f1:.4f} (support={supp})")

    print(f"\nConfusion matrix (rows=true, cols=predicted; order: {labels}):")
    print(matrix)

    print("\nConfidence threshold coverage:")
    for t in (0.5, 0.7, 0.8, 0.9):
        mask = conf >= t
        kept_acc = accuracy_score(y[mask], predicted[mask]) if mask.any() else float("nan")
        print(f"  threshold {t:.1f} -> accuracy on kept samples: {kept_acc:.4f} / coverage: {mask.mean():.4f}")

    return predicted, matrix, report


if not csv_path.is_file():
    fail(f"phrase landmark CSV not found: {csv_path}")

frame, raw_values, hands, numeric_stems, label_order = load_rows(csv_path)
content_hash = dataset_hash(csv_path)

label_to_index = {label: index for index, label in enumerate(label_order)}
all_features, usable_rows = make_features(hands, args.aspect)

if len(usable_rows) != len(frame):
    print(f"Dropped {len(frame) - len(usable_rows)} rows with degenerate landmarks during preprocessing")

frame = frame.iloc[usable_rows].reset_index(drop=True)
raw_values = raw_values[usable_rows]
numeric_stems = numeric_stems[usable_rows]
hands = [hands[i] for i in usable_rows]
y = frame["label"].map(label_to_index).to_numpy(dtype=np.int64)

# Create group labels combining class and frame span to avoid leakage
groups = np.asarray(
    [f"{lab}_{(max(0, stem - 1)) // args.group_span}" for lab, stem in zip(frame["label"], numeric_stems)]
)

manifest_path = out_dir / "split_manifest.json"
model_joblib_path = out_dir / "isl_phrases_mlp.joblib"

if args.evaluate_only:
    if not manifest_path.is_file() or not model_joblib_path.is_file():
        fail(f"saved phrase model or split manifest not found in {out_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["csv_sha256"] != content_hash:
        fail("phrase CSV differs from the CSV used to train this model; refusing an unmatched evaluation")
    val_idx = np.asarray(manifest["validation_indices"], dtype=np.int64)
    test_idx = np.asarray(manifest["test_indices"], dtype=np.int64)
    clf = joblib.load(model_joblib_path)

    print_metrics("Validation (Saved Split)", clf, all_features[val_idx], y[val_idx], label_order)
    print_metrics("Held-out Test (Saved Split)", clf, all_features[test_idx], y[test_idx], label_order)
else:
    if out_dir.exists() and not args.overwrite:
        fail(f"output directory already exists; use --overwrite to replace: {out_dir}")

    train_idx, val_idx, test_idx = stratified_grouped_split(
        frame, label_order, groups, args.test_size, args.validation_size, args.seed
    )

    train_hands = [hands[i] for i in train_idx]
    train_labels = y[train_idx]

    rng = np.random.default_rng(args.seed)
    augmented_features = []
    augmented_labels = []

    for sample_hands, sample_label in zip(train_hands, train_labels):
        variants = [sample_hands]
        if not args.no_mirror:
            variants.append(mirror(sample_hands))

        if args.augment > 0:
            extra_variants = []
            for base_variant in variants:
                for _ in range(args.augment):
                    extra_variants.append(augment_hands(base_variant, rng))
            variants.extend(extra_variants)

        for variant in variants:
            vector = build_features(
                variant,
                two_hand=TWO_HAND,
                rotate=ROTATE,
                aspect=args.aspect,
                extras=EXTRAS,
            )
            if vector is not None:
                augmented_features.append(vector)
                augmented_labels.append(sample_label)

    train_x = np.asarray(augmented_features, dtype=np.float32)
    train_y = np.asarray(augmented_labels, dtype=np.int64)

    print(
        f"\nDataset loaded: {len(frame)} rows across {len(label_order)} classes: {label_order}"
    )
    print(f"Feature vector dimension: {all_features.shape[1]}")
    print(
        f"Splits: train={len(train_idx)} (augmented to {len(train_x)} samples), "
        f"validation={len(val_idx)}, test={len(test_idx)}"
    )
    print(f"Total distinct groups: {len(np.unique(groups))}")

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layers),
        max_iter=400,
        early_stopping=True,
        random_state=args.seed,
    )
    clf.fit(train_x, train_y)

    val_pred, val_matrix, val_report = print_metrics(
        "Validation", clf, all_features[val_idx], y[val_idx], label_order
    )
    test_pred, test_matrix, test_report = print_metrics(
        "Held-out Test", clf, all_features[test_idx], y[test_idx], label_order
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    feature_settings = {
        "features": int(all_features.shape[1]),
        "two_hand": TWO_HAND,
        "rotate": ROTATE,
        "extras": EXTRAS,
        "aspect": round(args.aspect, 4),
        "threshold": 0.8,
        "note": "Features are produced by training/features.py build_features(). Layer math: out = activation(in @ W + b).",
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
    joblib.dump(clf, model_joblib_path)

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
    first_raw_hands = [hand.tolist() for hand in hands[first_test_index]]
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
    (out_dir / "prediction_test.json").write_text(json.dumps(prediction_test, indent=2), encoding="utf-8")
    (out_dir / "parity_test.json").write_text(json.dumps(prediction_test, indent=2), encoding="utf-8")

    prediction_path = out_dir / "test_predictions.csv"
    test_probabilities = clf.predict_proba(all_features[test_idx])
    with prediction_path.open("w", newline="", encoding="utf-8") as pred_file:
        writer = csv.writer(pred_file)
        writer.writerow(["image_path", "true_label", "predicted_label", *[f"p_{label}" for label in label_order]])
        for row_index, predicted, probabilities in zip(test_idx, test_pred, test_probabilities):
            writer.writerow([
                frame.iloc[row_index]["image_path"],
                label_order[int(y[row_index])],
                label_order[int(predicted)],
                *probabilities.tolist(),
            ])

    val_acc = float(accuracy_score(y[val_idx], val_pred))
    test_acc = float(accuracy_score(y[test_idx], test_pred))
    metrics_data = {
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "validation_report": val_report,
        "test_report": test_report,
        "validation_confusion_matrix": val_matrix.tolist(),
        "test_confusion_matrix": test_matrix.tolist(),
        "label_order": label_order,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    print(f"\nPhrase model training complete! Artifacts saved to: {out_dir}")
    print(f"Validation Accuracy: {val_acc:.4f} | Held-out Test Accuracy: {test_acc:.4f}")
