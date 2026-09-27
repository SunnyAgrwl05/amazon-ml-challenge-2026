#!/usr/bin/env python3
"""Fast XGBoost training using SQLite for target data."""
import csv, sqlite3, numpy as np, time, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("/Users/kumar/Desktop/amazon_ml_entity_resolution")
TRAIN = ROOT / "dataset" / "train"
DB_PATH = ROOT / "output" / "train_blocking.sqlite"
N_SAMPLES = 60_000

sys.path.insert(0, str(ROOT / "code"))
from business_entity_resolution.src.text_utils import norm_name, norm_address, norm_text
from business_entity_resolution.src.features import pair_features
from business_entity_resolution.src.model import Matcher
from business_entity_resolution.src.metrics import macro_f05

t0 = time.time()

# Load labels from cache
data = np.load(str(ROOT / "output" / "train_sample_20k.npy"), mmap_mode='r')
print(f"Cache pairs: {len(data):,}")

# Sample unique S1 entities (keep first occurrence of each S1)
seen = set()
samples = []
for i in range(min(N_SAMPLES, len(data))):
    sid = str(data[i]["s1"])
    tid = str(data[i]["tid"])
    if sid not in seen:
        seen.add(sid)
        samples.append((sid, tid, int(data[i]["label"])))
print(f"Sampled unique S1: {len(samples):,} in {time.time()-t0:.1f}s")

s1_set = {s for s, _, _ in samples}
tid_set = {t for _, t, _ in samples}

# Load S1 rows from TSV
print("Loading S1 rows...")
s1_map = {}
with open(TRAIN / "train_source1.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["entity_id"] in s1_set:
            s1_map[row["entity_id"]] = row
            if len(s1_map) == len(s1_set):
                break
print(f"  Loaded {len(s1_map):,}")

# Connect to DB and load target rows
print("Loading target rows from SQLite...")
con = sqlite3.connect(str(DB_PATH))
con.row_factory = sqlite3.Row
cur = con.cursor()

# Batch load targets
placeholders = ",".join("?" * len(tid_set))
query = f"SELECT entity_id, business_name, business_address, country FROM targets WHERE entity_id IN ({placeholders})"
cur.execute(query, list(tid_set))
target_map = {}
for row in cur.fetchall():
    target_map[row["entity_id"]] = dict(row)
con.close()
print(f"  Loaded {len(target_map):,} targets")

# Build features
print("Building features...")
X, y, groups = [], [], []
rows = []
for sid, tid, label in samples:
    s1_row = s1_map.get(sid)
    t_row = target_map.get(tid)
    if s1_row is None or t_row is None:
        continue
    try:
        feat = pair_features(s1_row, t_row)
    except Exception:
        continue
    X.append(feat)
    y.append(label)
    groups.append(sid)
    rows.append((sid, tid))

X = np.asarray(X)
y = np.asarray(y)
print(f"  X={X.shape}, positives={int(y.sum()):,} in {time.time()-t0:.1f}s")

# Train/val split
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr, va = next(splitter.split(X, y, groups=groups))
print(f"Train={len(tr):,} Val={len(va):,}")

# Train
print("Training...")
model = Matcher()
model.fit(X[tr], y[tr])

# Score
scores = model.predict_proba(X[va])
by_entity = defaultdict(list)
for (sid, eid), sc in zip([rows[i] for i in va], scores):
    by_entity[sid].append((eid, float(sc)))

truth_by_s1 = defaultdict(set)
for sid, tid, label in samples:
    if label == 1:
        truth_by_s1[sid].add(tid)
val_truth = {sid: truth_by_s1[sid] for sid in set(groups[i] for i in va)}

# Threshold sweep
thresholds = np.arange(0.10, 0.95, 0.01)
best_t, best_s = 0.50, -1.0
top5 = []
for t in thresholds:
    pred = {}
    for sid, pairs in by_entity.items():
        pred[sid] = {eid for eid, sc in pairs if sc >= t}
    s = macro_f05(val_truth, pred)
    if s > best_s:
        best_t, best_s = float(t), float(s)
    top5.append((t, s))
top5.sort(key=lambda x: -x[1])
total = time.time() - t0
print(f"\n{'='*60}")
print(f"TRAINING RESULT")
print(f"{'='*60}")
print(f"Sample S1           : {len(seen):,}")
print(f"Feature matrix      : {X.shape}")
print(f"Positive pairs      : {int(y.sum()):,}")
print(f"Val macro F0.5      : {best_s:.4f}  (threshold={best_t:.2f})")
print(f"\nTop-5 thresholds:")
for t, s in top5[:5]:
    print(f"  t={t:.2f}  F0.5={s:.4f}")
print(f"Total runtime       : {total:.1f}s")
