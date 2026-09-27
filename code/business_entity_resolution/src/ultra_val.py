#!/usr/bin/env python3
"""Ultra-fast F0.5 validation — single-connection, exact-name signal only."""
import csv, sqlite3, time, sys, numpy as np
from pathlib import Path

ROOT = Path("/Users/kumar/Desktop/amazon_ml_entity_resolution")
TRAIN = ROOT / "dataset" / "train"
DB_PATH = ROOT / "output" / "train_blocking.sqlite"
N = 1000

sys.path.insert(0, str(ROOT / "code"))
from business_entity_resolution.src.text_utils import norm_name, norm_address, norm_text

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

def macro_f05(truth, pred):
    vals = []
    for k in truth:
        t = truth[k]
        p = pred.get(k, set())
        if not t and not p: vals.append(1.0); continue
        if not t or not p: vals.append(0.0); continue
        tp = len(t & p)
        pr = tp/len(p); rc = tp/len(t)
        vals.append(0.0 if (pr==0 and rc==0) else 1.25*pr*rc/(0.25*pr+rc))
    return float(np.mean(vals)) if vals else 0.0

def main():
    s1 = load_s1(N)
    ids = set(s1.keys())
    gt = load_gt(ids)
    matched = [k for k,v in gt.items() if v]
    total_links = sum(len(v) for v in gt.values())
    print(f"S1: {len(s1)} matched: {len(matched)} links: {total_links}")

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = lambda c,r: r[0]
    cur = con.cursor()

    t0 = time.time()
    pred_all = {}   # predict all exact-name candidates
    covered_links = 0
    total_cands = 0
    max_cands = 0
    counts = []
    covered_ent = 0

    for sid, row in s1.items():
        nn = norm_name(row.get("business_name","") or "")
        na = norm_address(row.get("business_address","") or "")
        na_norm = norm_text(row.get("business_address","") or "")
        country = norm_text(row.get("country","") or "")

        cand = set()
        if nn:
            cur.execute("SELECT entity_id FROM targets WHERE norm_name=?", (nn,))
            cand.update(cur.fetchall())
        if na:
            cur.execute("SELECT entity_id FROM targets WHERE norm_address=?", (na,))
            cand.update(cur.fetchall())

        total_cands += len(cand)
        max_cands = max(max_cands, len(cand))
        counts.append(len(cand))
        true_ids = gt.get(sid, set())
        hit = true_ids & cand
        covered_links += len(hit)
        if hit: covered_ent += 1
        pred_all[sid] = cand

    elapsed = time.time() - t0
    con.close()

    counts.sort(); nn_ = len(counts)
    p50 = counts[nn_//2]
    p95 = counts[int(nn_*0.95)] if nn_>0 else 0

    tp = sum(len(gt[k] & pred_all[k]) for k in gt)
    pred_total = sum(len(pred_all[k]) for k in gt)
    prec = tp/pred_total if pred_total else 0
    rec = tp/total_links if total_links else 0
    f05 = macro_f05(gt, pred_all)

    print(f"\n{'='*60}")
    print(f"EXACT-NAME+ADDR BLOCKING — {N} S1 (train)")
    print(f"{'='*60}")
    print(f"S1 evaluated            : {len(s1):,}")
    print(f"Matched S1              : {len(matched):,}")
    print(f"True links              : {total_links:,}")
    print(f"Covered links           : {covered_links:,}  (LINK RECALL {covered_links/total_links*100:.2f}%)")
    print(f"Entity coverage         : {covered_ent}/{len(matched)} ({covered_ent/len(matched)*100:.2f}%)")
    print(f"Avg candidates/S1       : {total_cands/len(s1):.1f}")
    print(f"Median candidates/S1    : {p50}")
    print(f"P95 candidates/S1       : {p95}")
    print(f"Max candidates/S1       : {max_cands:,}")
    print(f"\nPrediction = ALL candidates:")
    print(f"Precision               : {prec:.4f}")
    print(f"Recall                  : {rec:.4f}")
    print(f"Entity macro F0.5       : {f05:.4f}")
    print(f"Runtime                 : {elapsed:.1f}s")

if __name__ == "__main__":
    main()
