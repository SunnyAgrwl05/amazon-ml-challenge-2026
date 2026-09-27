#!/usr/bin/env python3
"""Fast XGBoost training on blocking candidates from small S1 sample."""
import csv, sqlite3, time, sys, numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/Users/kumar/Desktop/amazon_ml_entity_resolution")
TRAIN = ROOT / "dataset" / "train"
DB_PATH = ROOT / "output" / "train_blocking.sqlite"
N = 2000  # S1 sample

sys.path.insert(0, str(ROOT / "code"))
from business_entity_resolution.src.text_utils import norm_name, norm_address, norm_text
from business_entity_resolution.src.features import pair_features
from business_entity_resolution.src.model import Matcher
from business_entity_resolution.src.metrics import macro_f05

main_start = time.time()

def load_s1(n):
    s1 = {}
    with open(TRAIN / "train_source1.tsv") as f:
        r = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(r):
            if i >= n: break
            s1[row["entity_id"]] = row
    return s1

def load_gt(ids):
    gt = {}
    with open(TRAIN / "train_ground_truth.tsv") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            sid = row["source1_entity_id"]
            if sid in ids:
                v = (row.get("matched_entity_ids") or "").strip()
                gt[sid] = {x.strip() for x in v.split(",") if x.strip()} if v else set()
    return gt

def load_candidate_targets(cand_ids):
    """Load only the target rows we need."""
    target_by_id = {}
    wanted = set(cand_ids)
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        with open(TRAIN / fn) as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                eid = row["entity_id"]
                if eid in wanted:
                    target_by_id[eid] = row
                    if len(target_by_id) == len(wanted):
                        return target_by_id
    return target_by_id

def main():
    s1 = load_s1(N)
    ids = set(s1.keys())
    gt = load_gt(ids)
    matched = [k for k,v in gt.items() if v]
    total_links = sum(len(v) for v in gt.values())
    print(f"S1: {len(s1)} matched: {len(matched)} links: {total_links}")

    # Get candidates from DB (single connection)
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = lambda c,r: r[0]
    cur = con.cursor()

    t0 = time.time()
    print("Getting candidates...")
    pairs = []  # (s1_id, s1_row, tid, label)
    all_cand_ids = set()
    for sid, row in s1.items():
        nn = norm_name(row.get("business_name","") or "")
        na = norm_address(row.get("business_address","") or "")
        country = norm_text(row.get("country","") or "")

        cand = set()
        if nn:
            cur.execute("SELECT entity_id FROM targets WHERE norm_name=?", (nn,))
            cand.update(cur.fetchall())
        if na:
            cur.execute("SELECT entity_id FROM targets WHERE norm_address=?", (na,))
            cand.update(cur.fetchall())
        if nn and country:
            cur.execute("SELECT entity_id FROM targets WHERE country_norm=? AND norm_name=?", (country, nn))
            cand.update(cur.fetchall())
        if na and country:
            cur.execute("SELECT entity_id FROM targets WHERE country_norm=? AND norm_address=?", (country, na))
            cand.update(cur.fetchall())
        if nn and len(nn)>=3:
            cur.execute("SELECT entity_id FROM targets WHERE name_prefix=? LIMIT 2000", (nn[:3],))
            cand.update(cur.fetchall())

        all_cand_ids.update(cand)
        true_ids = gt.get(sid, set())
        for tid in cand:
            pairs.append((sid, row, tid, 1 if tid in true_ids else 0))

    elapsed = time.time() - t0
    con.close()
    print(f"  Pairs: {len(pairs):,} unique targets: {len(all_cand_ids):,} ({elapsed:.1f}s)")

    # Load only needed targets
    print("Loading target rows...")
    t0 = time.time()
    target_by_id = load_candidate_targets(all_cand_ids)
    print(f"  Loaded {len(target_by_id):,} targets ({time.time()-t0:.1f}s)")

    # Build features
    print("Building features...")
    t0 = time.time()
    X, y, groups, rows = [], [], [], []
    for sid, s1_row, tid, label in pairs:
        trow = target_by_id.get(tid)
        if trow is None:
            continue
        try:
            feat = pair_features(s1_row, trow)
        except Exception:
            continue
        X.append(feat)
        y.append(label)
        groups.append(sid)
        rows.append((sid, tid))
    X = np.asarray(X)
    y = np.asarray(y)
    print(f"  X: {X.shape}, positives: {int(y.sum()):,} ({time.time()-t0:.1f}s)")

    # Train/val split by S1 entity
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, va = next(splitter.split(X, y, groups=groups))
    print(f"Train: {len(tr):,} Val: {len(va):,}")

    # Train
    print("Training XGBoost...")
    t0 = time.time()
    model = Matcher()
    model.fit(X[tr], y[tr])
    print(f"  Trained in {time.time()-t0:.1f}s")

    # Score on val
    scores = model.predict_proba(X[va])
    by_entity = defaultdict(list)
    for (sid, eid), sc in zip([rows[i] for i in va], scores):
        by_entity[sid].append((eid, float(sc)))

    val_truth = {sid: gt[sid] for sid in set(groups[i] for i in va)}

    # Threshold sweep
    thresholds = np.arange(0.10, 0.95, 0.01)
    best_t, best_s = 0.50, -1.0
    results = []
    for t in thresholds:
        pred = {}
        for sid, pairs in by_entity.items():
            pred[sid] = {eid for eid, sc in pairs if sc >= t}
        s = macro_f05(val_truth, pred)
        results.append((t, s))
        if s > best_s:
            best_t, best_s = float(t), float(s)

    # Top 5 thresholds
    results.sort(key=lambda x: -x[1])
    print(f"\n{'='*60}")
    print(f"TRAINING ON {N} S1 SAMPLE")
    print(f"{'='*60}")
    print(f"Candidate pairs     : {len(pairs):,}")
    print(f"Feature matrix      : {X.shape}")
    print(f"Positive pairs      : {int(y.sum()):,}")
    print(f"Val macro F0.5      : {best_s:.4f}  (threshold={best_t:.2f})")
    print(f"\nTop-5 thresholds:")
    for t, s in results[:5]:
        print(f"  t={t:.2f}  F0.5={s:.4f}")
    print(f"\nTraining time       : {time.time()-main_start:.1f}s total")

if __name__ == "__main__":
    main()
