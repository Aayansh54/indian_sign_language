"""
train_clean.py - trains on data/landmarks.csv with the cleaning step, mirror augmentation and an HONEST split.

Run from your project folder (D:\\ISL):
    python train_clean.py                       # the 6 one-handed letters C I L O U V, single-hand mode
    python train_clean.py --letters ABCDEFGHIJKLMNOPQRSTUVWXYZ --two-hand
    python train_clean.py --test-csv data/test_landmarks.csv   # also score an independent test set
Outputs go to model/ : isl_clean_mlp.joblib, model.json, labels.json, config.json, parity_test.json
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from features import hands_from_flat, mirror, build_features, augment_hands

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/landmarks.csv")
ap.add_argument("--letters", default="CILOUV", help="letters to train on, e.g. CILOUV or ABC...Z")
ap.add_argument("--two-hand", action="store_true", help="use both hands (131 features)")
ap.add_argument("--no-rotate", action="store_true", help="skip the rotate step (compare both!)")
ap.add_argument("--extras", action="store_true", help="add 15 fingertip distance features (helps O vs C)")
ap.add_argument("--augment", type=int, default=0, help="extra random tilted/jittered copies per training sample (try 2)")
ap.add_argument("--no-mirror", action="store_true", help="skip mirror augmentation")
ap.add_argument("--train-aspect", type=float, default=1.0, help="width/height of the training images")
ap.add_argument("--extra-csv", nargs="*", default=[], help="more landmark CSVs to add to training, e.g. data/own_landmarks.csv")
ap.add_argument("--test-csv", default=None, help="landmarks CSV from images/people NOT used for training")
ap.add_argument("--out", default="model")
a = ap.parse_args()
rotate, letters = not a.no_rotate, sorted(set(a.letters.upper()))
CLS = np.array(letters)                      # class number -> letter (numbers train more reliably than text)
IDX = {c: i for i, c in enumerate(letters)}


def load(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    if len(cols) != 128 or cols[:2] != ["label", "image_path"] or cols[2] != "h0_p0_x" or cols[5] != "h0_p1_x":
        sys.exit(f"{path}: unexpected columns {cols[:6]}... expected label,image_path,h0_p0_x,h0_p0_y,h0_p0_z,h0_p1_x ...")
    df = df[df.label.isin(letters)].reset_index(drop=True)
    vals = df.iloc[:, 2:].to_numpy(dtype=np.float32)
    hands = [hands_from_flat(r) for r in vals]
    # neighbouring frames look alike, so keep blocks of 30 consecutive rows of a letter together in one split
    seen, groups = {}, []
    for lab in df.label:
        k = seen.get(lab, 0)
        seen[lab] = k + 1
        groups.append(f"{Path(path).stem}_{lab}_{k // 30}")
    return df.label.to_numpy(), hands, np.array(groups)


rng = np.random.default_rng(0)


def featurize(labels, hands, augment):
    X, y = [], []
    for lab, hs in zip(labels, hands):
        variants = [hs] + ([mirror(hs)] if augment and not a.no_mirror else [])
        if augment and a.augment > 0:
            variants += [augment_hands(v, rng) for v in list(variants) for _ in range(a.augment)]
        for v in variants:
            f = build_features(v, a.two_hand, rotate, a.train_aspect, a.extras)
            if f is not None:
                X.append(f); y.append(IDX[lab])
    return np.array(X, dtype=np.float32), np.array(y)


def report(name, clf, X, y):
    proba = clf.predict_proba(X)
    pred = clf.classes_[proba.argmax(1)]
    conf = proba.max(1)
    print(f"\n===== {name}: {len(y)} samples =====")
    print("accuracy:", round(accuracy_score(y, pred), 4))
    print(classification_report(y, pred, labels=clf.classes_, target_names=list(CLS[clf.classes_]), zero_division=0))
    print("confusion matrix (rows=true, cols=predicted), order:", [str(c) for c in CLS])
    cm = confusion_matrix(y, pred, labels=clf.classes_)
    print(cm)
    pairs = sorted(((int(cm[i, j]), CLS[clf.classes_[i]], CLS[clf.classes_[j]])
                    for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0), reverse=True)
    print("MOST COMMON MIX-UPS (count: true letter -> predicted letter):", [f"{n}: {t}->{p}" for n, t, p in pairs[:12]])
    rec = {CLS[clf.classes_[i]]: cm[i, i] / cm[i].sum() for i in range(len(cm)) if cm[i].sum() > 0}
    print("WEAKEST LETTERS (share recognised):", [f"{k} {v:.2f}" for k, v in sorted(rec.items(), key=lambda kv: kv[1])[:8]])
    print("confidence threshold -> accuracy on kept samples / share of samples kept")
    for t in (0.5, 0.7, 0.8, 0.9):
        k = conf >= t
        print(f"  {t:.1f} -> {accuracy_score(y[k], pred[k]) if k.any() else float('nan'):.3f} / {k.mean():.3f}")


labels, hands, groups = load(a.csv)
for extra in a.extra_csv:
    l2, h2, g2 = load(extra)
    print(f"added {len(l2)} rows from {extra}")
    labels, hands, groups = np.concatenate([labels, l2]), hands + h2, np.concatenate([groups, g2])
print(f"loaded {len(labels)} rows for letters {''.join(letters)} | rows per letter:", {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})
def split_by_group(labels, groups, test_size=0.2, seed=42):
    """hold out ~20% of the blocks of EVERY letter, so every letter gets validation samples"""
    rng = np.random.default_rng(seed)
    val = np.zeros(len(labels), dtype=bool)
    for lab in np.unique(labels):
        gs = np.unique(groups[labels == lab])
        rng.shuffle(gs)
        k = max(1, int(round(len(gs) * test_size))) if len(gs) > 1 else 0
        val |= np.isin(groups, gs[:k])
    return np.where(~val)[0], np.where(val)[0]


tr, va = split_by_group(labels, groups)
Xtr, ytr = featurize(labels[tr], [hands[i] for i in tr], augment=True)
Xva, yva = featurize(labels[va], [hands[i] for i in va], augment=False)
missing = sorted(set(letters) - set(CLS[yva]))
if missing:
    print("WARNING: no validation samples for letters:", missing)
print("train samples (with mirror):", len(ytr), "| validation samples:", len(yva), "| features:", Xtr.shape[1])

clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=True, random_state=42)
clf.fit(Xtr, ytr)
report("VALIDATION (grouped split, still the same people/images as training)", clf, Xva, yva)

if a.test_csv:
    tl, th, _ = load(a.test_csv)
    Xte, yte = featurize(tl, th, augment=False)
    report("INDEPENDENT TEST  <- this is your real number", clf, Xte, yte)
else:
    print("\nNOTE: no --test-csv given. Validation accuracy is still optimistic. Extract landmarks from data/Testing "
          "(and teammates' webcam photos) and pass them with --test-csv for the real number.")

# ------------------------------------------------------------ export for the website
out = Path(a.out); out.mkdir(exist_ok=True)
if len(letters) < 3:
    sys.exit("need at least 3 letters for the browser export format")
n = len(clf.coefs_)
layers = [{"W": np.round(W, 6).tolist(), "b": np.round(b, 6).tolist(), "activation": "relu" if i < n - 1 else "softmax"}
          for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_))]
cfg = {"features": int(Xtr.shape[1]), "two_hand": a.two_hand, "rotate": rotate, "extras": a.extras, "threshold": 0.8,
       "note": "layer math: out = activation(in @ W + b). Features come from features.py build_features()."}
(out / "model.json").write_text(json.dumps({"layers": layers}))
(out / "labels.json").write_text(json.dumps([str(c) for c in CLS[clf.classes_]]))
(out / "config.json").write_text(json.dumps(cfg, indent=1))
joblib.dump(clf, out / "isl_clean_mlp.joblib")
ex = hands[va[0]]
fx = build_features(ex, a.two_hand, rotate, a.train_aspect, a.extras)
(out / "parity_test.json").write_text(json.dumps({
    "aspect": a.train_aspect, "raw_hands": [h.tolist() for h in ex],
    "expected_features": fx.tolist(), "expected_probabilities": clf.predict_proba(fx.reshape(1, -1))[0].tolist(),
    "labels": [str(c) for c in CLS[clf.classes_]]}))
print(f"\nsaved to {out}/ : model.json labels.json config.json isl_clean_mlp.joblib parity_test.json")
