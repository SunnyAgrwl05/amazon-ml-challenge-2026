from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .blocking import Blocker
from .features import pair_features, FEATURE_NAMES
from .model import Matcher
from .data import load_ground_truth
from .metrics import tune_threshold, macro_f05

def make_pairs(s1, targets, truth, blocker):
    X, y, groups = [], [], []
    rows = []
    for _, a in s1.iterrows():
        sid = a.entity_id
        cand_idx = blocker.candidates(a)
        positives = truth.get(sid, set())
        for j in cand_idx:
            b = targets.iloc[j]
            label = int(b.entity_id in positives)
            X.append(pair_features(a, b))
            y.append(label)
            groups.append(sid)
            rows.append((sid, b.entity_id, label))
    return np.asarray(X), np.asarray(y), groups, rows

def main(base):
    base = Path(base)
    train = base / "dataset" / "train"
    s1 = pd.read_csv(train / "train_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(train / "train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(train / "train_source3.tsv", sep="\t", dtype=str).fillna("")
    truth = load_ground_truth(train / "train_ground_truth.tsv")

    targets = pd.concat([s2, s3], ignore_index=True)
    blocker = Blocker(targets)

    X, y, groups, rows = make_pairs(s1, targets, truth, blocker)

    # Split by Source-1 entity to prevent pair-level leakage.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, va = next(splitter.split(X, y, groups=groups))

    model = Matcher().fit(X[tr], y[tr])

    scores = model.predict_proba(X[va])
    by_entity = {}
    for (sid, eid, _), score in zip([rows[i] for i in va], scores):
        by_entity.setdefault(sid, []).append((eid, float(score)))

    val_truth = {sid: truth[sid] for sid in set(groups[i] for i in va)}
    threshold, score = tune_threshold(by_entity, val_truth)

    out = base / "output"
    out.mkdir(exist_ok=True)
    model.save(out / "matcher.joblib")
    (out / "threshold.txt").write_text(f"{threshold:.4f}\n")
    print(f"Validation macro F0.5: {score:.6f}")
    print(f"Chosen threshold: {threshold:.4f}")
    print(f"Pairs: {len(y):,}; positives: {int(y.sum()):,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    args = parser.parse_args()
    main(args.base)
