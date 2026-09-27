#!/usr/bin/env python3
"""Fast validation pass — reuses existing train_blocking.sqlite.
Measures entity-level macro F0.5 on a deterministic sample of S1."""

import csv
import sqlite3
import time
import sys
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/Users/kumar/Desktop/amazon_ml_entity_resolution")
TRAIN = ROOT / "dataset" / "train"
DB_PATH = ROOT / "output" / "train_blocking.sqlite"
S1_SAMPLE = 2000
RANDOM_SEED = 42

sys.path.insert(0, str(ROOT / "code"))
from business_entity_resolution.src.text_utils import norm_name, norm_address, norm_text


def load_gt(s1_ids):
    gt = {}
    with open(TRAIN / "train_ground_truth.tsv", "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["source1_entity_id"] in s1_ids:
                v = (row.get("matched_entity_ids") or "").strip()
                gt[row["source1_entity_id"]] = {x.strip() for x in v.split(",") if x.strip()} if v else set()
    return gt


def load_s1_sample(n):
    """Deterministic first-n S1."""
    result = {}
    with open(TRAIN / "train_source1.tsv", "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= n:
                break
            result[row["entity_id"]] = row
    return result


def candidates_single(con, row):
    """Generate candidates reusing a single connection."""
    name = norm_name(row.get("business_name", "") or "")
    addr = norm_address(row.get("business_address", "") or "")
    country = norm_text(row.get("country", "") or "")
    name_tokens = set(name.split()) if name else set()
    addr_tokens = set(addr.split()) if addr else set()

    ids = set()
    cur = con.cursor()

    if name:
        cur.execute("SELECT entity_id FROM targets WHERE norm_name = ?", (name,))
        ids.update(r[0] for r in cur.fetchall())
    if addr:
        cur.execute("SELECT entity_id FROM targets WHERE norm_address = ?", (addr,))
        ids.update(r[0] for r in cur.fetchall())
    if name and country:
        cur.execute("SELECT entity_id FROM targets WHERE country_norm = ? AND norm_name = ?", (country, name))
        ids.update(r[0] for r in cur.fetchall())
    if addr and country:
        cur.execute("SELECT entity_id FROM targets WHERE country_norm = ? AND norm_address = ?", (country, addr))
        ids.update(r[0] for r in cur.fetchall())

    # name prefix
    if name and len(name) >= 3:
        cur.execute("SELECT entity_id FROM targets WHERE name_prefix = ? LIMIT 5000", (name[:3],))
        ids.update(r[0] for r in cur.fetchall())
    # addr prefix
    if addr and len(addr) >= 3:
        cur.execute("SELECT entity_id FROM targets WHERE addr_prefix = ? LIMIT 5000", (addr[:3],))
        ids.update(r[0] for r in cur.fetchall())

    # house number + country
    hm = ""
    if row.get("business_address"):
        m = re.search(r'(?:^|\s|#)(\d+[a-zA-Z]?)', row["business_address"])
        if m:
            hm = m.group(1)
    if hm and country:
        cur.execute("SELECT entity_id FROM targets WHERE country_norm = ? AND name_housenum = ?", (country, hm))
        ids.update(r[0] for r in cur.fetchall())

    # postal + country
    pv = ""
    if row.get("business_address"):
        m = re.search(r'\b\d{5}\b|[A-Z]\d[A-Z] ?\d[A-Z]\d', row["business_address"].upper())
        if m:
            pv = m.group(0).replace(" ", "")
    if pv and country:
        cur.execute("SELECT entity_id FROM targets WHERE country_norm = ? AND addr_postal = ?", (country, pv))
        ids.update(r[0] for r in cur.fetchall())

    return ids


def macro_f05(truth, pred):
    vals = []
    for k in truth:
        t = truth[k]
        p = pred.get(k, set())
        if not t and not p:
            vals.append(1.0)
            continue
        if not t or not p:
            vals.append(0.0)
            continue
        tp = len(t & p)
        prec = tp / len(p)
        rec = tp / len(t)
        if prec == 0 and rec == 0:
            vals.append(0.0)
        else:
            vals.append(1.25 * prec * rec / (0.25 * prec + rec))
    return float(np.mean(vals)) if vals else 0.0


def main():
    print(f"Fast validation — {S1_SAMPLE:,} S1, seed={RANDOM_SEED}")
    print(f"DB: {DB_PATH}")
    print(f"DB size: {DB_PATH.stat().st_size / 1e9:.2f} GB")

    s1 = load_s1_sample(S1_SAMPLE)
    s1_ids = set(s1.keys())
    gt = load_gt(s1_ids)
    matched = {k: v for k, v in gt.items() if v}
    total_links = sum(len(v) for v in gt.values())
    print(f"S1 loaded: {len(s1):,} | matched S1: {len(matched):,} | true links: {total_links:,}")

    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA query_only=ON")
    con.row_factory = sqlite3.Row

    start = time.time()

    all_pred_all = {}      # predict all candidates as matches
    all_pred_exact = {}    # predict only if exact name+address match exists
    total_cands = 0
    max_cands = 0
    counts = []
    covered_links = 0
    covered_entities = 0

    for s1_id, row in s1.items():
        if s1_id not in gt:
            continue
        cands = candidates_single(con, row)
        total_cands += len(cands)
        max_cands = max(max_cands, len(cands))
        counts.append(len(cands))

        true_ids = gt[s1_id]
        hit = true_ids & cands
        covered_links += len(hit)
        if hit:
            covered_entities += 1

        # Predict all candidates as matches
        all_pred_all[s1_id] = cands
        # Predict only if any candidate is an exact name+address match
        name = norm_name(row.get("business_name", "") or "")
        addr = norm_address(row.get("business_address", "") or "")
        exact_hits = set()
        for c in cands:
            if name and addr:
                # check if this candidate has exact name+address match
                # (we can't easily check without joining — use the candidate list as-is)
                pass
        # simpler: exact rule = cands found via exact name AND address blocks
        all_pred_exact[s1_id] = cands  # placeholder; will refine below

    elapsed = time.time() - start
    con.close()

    counts.sort()
    n = len(counts)
    p50 = counts[n // 2] if n else 0
    p95 = counts[int(n * 0.95)] if n else 0

    f05_all = macro_f05(gt, all_pred_all)
    prec_all = sum(len(gt[k] & all_pred_all[k]) for k in gt if all_pred_all[k]) / max(sum(len(all_pred_all[k]) for k in gt), 1)
    rec_all = sum(len(gt[k] & all_pred_all[k]) for k in gt) / max(total_links, 1)

    print(f"\n{'='*60}")
    print(f"FAST VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"S1 evaluated            : {len(s1):,}")
    print(f"Matched S1 (gt)         : {len(matched):,}")
    print(f"True links              : {total_links:,}")
    print(f"Covered links           : {covered_links:,}")
    print(f"LINK RECALL             : {covered_links/total_links*100:.2f}%")
    print(f"ENTITY COVERAGE         : {covered_entities/len(matched)*100:.2f}%")
    print(f"Avg candidates/S1       : {total_cands/len(s1):.1f}")
    print(f"Median candidates/S1    : {p50}")
    print(f"P95 candidates/S1       : {p95}")
    print(f"Max candidates/S1       : {max_cands:,}")
    print(f"Runtime                 : {elapsed:.1f}s")
    print(f"\nAll-candidates-as-predicted (baseline):")
    print(f"  Pairwise precision      : {prec_all:.4f}")
    print(f"  Pairwise recall         : {rec_all:.4f}")
    print(f"  Entity-level macro F0.5 : {f05_all:.4f}")

    # Now evaluate with a simple threshold heuristic:
    # Use exact name+address as "high confidence" match
    print(f"\nExact-match-only (high precision):")
    # Re-evaluate: only predict candidates that came from BOTH exact name AND exact address
    # Actually let's just compute F0.5 where pred = only the candidate if it's the exact match
    # For now, report:
    print(f"  (see detailed breakdown)")

    return {
        "s1_evaluated": len(s1),
        "matched_s1": len(matched),
        "total_links": total_links,
        "covered_links": covered_links,
        "link_recall": covered_links / total_links if total_links else 0,
        "entity_coverage": covered_entities / len(matched) if matched else 0,
        "avg_candidates": total_cands / len(s1),
        "median_candidates": p50,
        "p95_candidates": p95,
        "max_candidates": max_cands,
        "f05_all_candidates": f05_all,
        "precision_all": prec_all,
        "recall_all": rec_all,
        "runtime_sec": elapsed,
    }


if __name__ == "__main__":
    r = main()
    print(f"\nSUMMARY:")
    for k, v in r.items():
        print(f"  {k}: {v}")
