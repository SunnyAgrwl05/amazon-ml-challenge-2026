#!/usr/bin/env python3

import csv
import sqlite3
import time
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path("/Users/kumar/Desktop/amazon_ml_entity_resolution")
TRAIN = ROOT / "dataset" / "train"
DB_PATH = ROOT / "output" / "train_blocking.sqlite"
CACHE = ROOT / "output" / "train_sample_20k.npy"

N_NEG_PER_S1 = 20
RANDOM_SEED = 42

sys.path.insert(0, str(ROOT / "code"))

from business_entity_resolution.src.features import pair_features
from business_entity_resolution.src.model import Matcher
from business_entity_resolution.src.metrics import macro_f05


t0 = time.time()
rng = np.random.default_rng(RANDOM_SEED)

print("=" * 70)
print("SAMPLED ENTITY-RESOLUTION TRAINING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Read candidate cache and collect positives + negatives
# ------------------------------------------------------------
print("\n[1/6] Reading candidate cache...")

data = np.load(CACHE, mmap_mode="r")

# Group row indices by S1.
# 20k S1 only -> dictionary is manageable.
groups = defaultdict(list)

for start in range(0, len(data), 2_000_000):
    end = min(start + 2_000_000, len(data))
    chunk = data[start:end]

    for sid in np.unique(chunk["s1"]):
        mask = chunk["s1"] == sid
        idxs = np.flatnonzero(mask) + start
        groups[str(sid)].extend(idxs.tolist())

    print(f"  scanned {end:,}/{len(data):,}")

print(f"S1 entities: {len(groups):,}")

# ------------------------------------------------------------
# 2. Keep every positive and sample negatives per S1
# ------------------------------------------------------------
selected = []

for n, (sid, idxs) in enumerate(groups.items(), 1):

    pos = []
    neg = []

    for idx in idxs:
        if int(data[idx]["label"]) == 1:
            pos.append(idx)
        else:
            neg.append(idx)

    # Keep ALL positives.
    for idx in pos:
        selected.append(
            (
                sid,
                str(data[idx]["tid"]),
                1
            )
        )

    # Random negative sample.
    if len(neg) > N_NEG_PER_S1:
        chosen = rng.choice(
            np.asarray(neg),
            size=N_NEG_PER_S1,
            replace=False
        )
    else:
        chosen = neg

    for idx in chosen:
        selected.append(
            (
                sid,
                str(data[idx]["tid"]),
                0
            )
        )

print(f"Selected pairs: {len(selected):,}")
print(f"Selected positives: {sum(x[2] for x in selected):,}")
print(f"Selected negatives: {sum(x[2] == 0 for x in selected):,}")

# ------------------------------------------------------------
# 3. Load S1 rows
# ------------------------------------------------------------
print("\n[2/6] Loading S1 rows...")

s1_ids = {x[0] for x in selected}
s1_map = {}

with open(TRAIN / "train_source1.tsv", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        eid = row["entity_id"]

        if eid in s1_ids:
            s1_map[eid] = row

        if len(s1_map) == len(s1_ids):
            break

print(f"Loaded S1: {len(s1_map):,}")

# ------------------------------------------------------------
# 4. Load target rows from SQLite
# ------------------------------------------------------------
print("\n[3/6] Loading target rows...")

tid_ids = {x[1] for x in selected}

con = sqlite3.connect(str(DB_PATH))
con.row_factory = sqlite3.Row
cur = con.cursor()

target_map = {}

ids = list(tid_ids)

for start in range(0, len(ids), 5000):

    batch = ids[start:start + 5000]
    placeholders = ",".join("?" * len(batch))

    q = f"""
        SELECT
            entity_id,
            business_name,
            business_address,
            country
        FROM targets
        WHERE entity_id IN ({placeholders})
    """

    cur.execute(q, batch)

    for row in cur.fetchall():
        target_map[row["entity_id"]] = dict(row)

    print(f"  targets {min(start + 5000, len(ids)):,}/{len(ids):,}")

con.close()

print(f"Loaded targets: {len(target_map):,}")

# ------------------------------------------------------------
# 5. Build features
# ------------------------------------------------------------
print("\n[4/6] Building features...")

X = []
y = []
rows = []
entity_groups = []

for i, (sid, tid, label) in enumerate(selected, 1):

    a = s1_map.get(sid)
    b = target_map.get(tid)

    if a is None or b is None:
        continue

    try:
        feat = pair_features(a, b)
    except Exception:
        continue

    X.append(feat)
    y.append(label)
    rows.append((sid, tid))
    entity_groups.append(sid)

    if i % 25000 == 0:
        print(f"  features: {i:,}/{len(selected):,}")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int8)

print(f"Feature matrix: {X.shape}")
print(f"Positive: {int(y.sum()):,}")
print(f"Negative: {int((y == 0).sum()):,}")

# ------------------------------------------------------------
# 6. Group split + train + threshold
# ------------------------------------------------------------
print("\n[5/6] Train/validation split...")

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

tr, va = next(
    splitter.split(
        X,
        y,
        groups=entity_groups
    )
)

print(f"Train pairs: {len(tr):,}")
print(f"Val pairs:   {len(va):,}")

print("\n[6/6] Training model...")

model = Matcher()
model.fit(X[tr], y[tr])

scores = model.predict_proba(X[va])

by_entity = defaultdict(list)

for idx, score in zip(va, scores):
    sid, tid = rows[idx]
    by_entity[sid].append(
        (tid, float(score))
    )

# IMPORTANT:
# truth must come from ALL selected positives,
# not only validation-positive rows.
truth_by_s1 = defaultdict(set)

for sid, tid, label in selected:
    if label == 1:
        truth_by_s1[sid].add(tid)

val_entities = set(entity_groups[i] for i in va)

val_truth = {
    sid: truth_by_s1[sid]
    for sid in val_entities
}

print("\nThreshold sweep...")

results = []

for threshold in np.arange(0.50, 0.991, 0.01):

    pred = {}

    for sid, pairs in by_entity.items():
        pred[sid] = {
            tid
            for tid, score in pairs
            if score >= threshold
        }

    score = macro_f05(
        val_truth,
        pred
    )

    results.append(
        (float(threshold), float(score))
    )

results.sort(
    key=lambda x: x[1],
    reverse=True
)

best_t, best_score = results[0]

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"Total selected pairs : {len(selected):,}")
print(f"Feature matrix       : {X.shape}")
print(f"Positive pairs       : {int(y.sum()):,}")
print(f"Validation entities  : {len(val_entities):,}")
print(f"BEST F0.5            : {best_score:.6f}")
print(f"BEST THRESHOLD       : {best_t:.2f}")

print("\nTop 10 thresholds:")

for t, s in results[:10]:
    print(f"  threshold={t:.2f}  F0.5={s:.6f}")

# Save model + threshold
import joblib

MODEL_PATH = ROOT / "output" / "matcher.joblib"
THRESHOLD_PATH = ROOT / "output" / "threshold.txt"

joblib.dump(model, MODEL_PATH)

THRESHOLD_PATH.write_text(
    f"{best_t:.6f}\n",
    encoding="utf-8"
)

print("\nSaved:")
print(MODEL_PATH)
print(THRESHOLD_PATH)

print(f"\nRuntime: {time.time() - t0:.1f}s")
