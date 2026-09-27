#!/usr/bin/env python3
"""
Deterministic 20,000-S1 smoke-test diagnostic for SQLiteBlocker.
Reports:
- number of sampled S1 entities
- true match links
- covered true links
- LINK RECALL %
- S1 ENTITY COVERAGE %
- average candidates/S1
- median candidates/S1
- p95 candidates/S1
- maximum candidates/S1
- runtime
- SQLite database size
- approximate peak RAM
"""

import csv
import os
import sys
import time
import tracemalloc
import resource
from typing import Dict, Set, List

ROOT = "/Users/kumar/Desktop/amazon_ml_entity_resolution"
TRAIN = os.path.join(ROOT, "dataset", "train")
DB_PATH = os.path.join(ROOT, "output", "smoke_test_blocking.sqlite")

SAMPLE_SIZE = 20000
TARGET_TSV_PATHS = [
    os.path.join(TRAIN, "train_source2.tsv"),
    os.path.join(TRAIN, "train_source3.tsv")
]

sys.path.insert(0, os.path.join(ROOT, 'code'))

from business_entity_resolution.src.blocking import SQLiteBlocker


def load_s1_entities(sample_size: int) -> Dict[str, tuple]:
    """Load first `sample_size` entities from train_source1.tsv"""
    path = os.path.join(TRAIN, "train_source1.tsv")
    result = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
            result[row["entity_id"]] = (
                row.get("business_name", ""),
                row.get("business_address", ""),
                row.get("country", "")
            )

    return result


def load_ground_truth(s1_ids: Set[str]) -> Dict[str, Set[str]]:
    """Load ground truth matches for given S1 entity IDs."""
    path = os.path.join(TRAIN, "train_ground_truth.tsv")
    result = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            if s1_id in s1_ids:
                value = (row.get("matched_entity_ids") or "").strip()
                if value:
                    true_ids = {
                        x.strip() for x in value.split(",") if x.strip()
                    }
                    result[s1_id] = true_ids
                else:
                    result[s1_id] = set()

    return result


def get_db_size(path: str) -> int:
    """Get database file size in bytes."""
    return os.path.getsize(path) if os.path.exists(path) else 0


def get_peak_rss_mb() -> float:
    """Get peak resident set size in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0  # Linux returns KB, macOS returns bytes? Let's handle both


def run_smoke_test():
    print("=" * 60)
    print("SMOKE TEST: 20,000 S1 ENTITIES")
    print("=" * 60)
    print(f"Sample size: {SAMPLE_SIZE:,}")
    print(f"Target sources: train_source2.tsv, train_source3.tsv")
    print(f"Database: {DB_PATH}")
    print()

    # Start memory tracking
    tracemalloc.start()

    start_time = time.time()

    # Load S1 entities (sample)
    print(f"Loading {SAMPLE_SIZE:,} S1 entities...")
    s1_entities = load_s1_entities(SAMPLE_SIZE)
    s1_ids = set(s1_entities.keys())
    print(f"  Loaded: {len(s1_entities):,}")

    # Load ground truth for these S1 entities
    print("Loading ground truth...")
    ground_truth = load_ground_truth(s1_ids)
    total_true_links = sum(len(v) for v in ground_truth.values())
    matched_entities = len([k for k, v in ground_truth.items() if v])
    print(f"  Matched S1 entities: {matched_entities:,}")
    print(f"  True match links: {total_true_links:,}")

    # Build/load blocker
    print("\nBuilding SQLiteBlocker index...")
    block_start = time.time()

    # Remove existing database for clean test
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    blocker = SQLiteBlocker(
        target_tsv_paths=TARGET_TSV_PATHS,
        db_path=DB_PATH,
        name_prefix_len=3,
        addr_prefix_len=3,
        rare_token_min_freq=2,
        rare_token_max_freq=100,
        prefix_candidates_limit=5000
    )

    block_time = time.time() - block_start
    print(f"  Blocking index built in {block_time:.1f}s")

    # Generate candidates and evaluate
    print("\nGenerating candidates and evaluating...")
    eval_start = time.time()

    covered_links = 0
    covered_entities = 0
    total_candidates = 0
    max_candidates = 0
    candidate_counts = []
    processed = 0

    for s1_id in s1_ids:
        if s1_id not in ground_truth:
            continue

        name, address, country = s1_entities[s1_id]
        true_ids = ground_truth[s1_id]

        row_dict = {
            "entity_id": s1_id,
            "business_name": name,
            "business_address": address,
            "country": country
        }

        candidates = blocker.candidates(row_dict)

        total_candidates += len(candidates)
        candidate_counts.append(len(candidates))
        if len(candidates) > max_candidates:
            max_candidates = len(candidates)

        hit = true_ids.intersection(candidates)
        covered_links += len(hit)

        if hit:
            covered_entities += 1

        processed += 1
        if processed % 5000 == 0:
            elapsed = time.time() - eval_start
            avg_cand = total_candidates / processed if processed > 0 else 0
            print(f"  Processed {processed:,} / {matched_entities:,} | "
                  f"avg candidates: {avg_cand:.1f} | "
                  f"max: {max_candidates} | "
                  f"elapsed: {elapsed:.1f}s")

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time

    # Calculate percentiles
    candidate_counts.sort()
    n = len(candidate_counts)
    p50 = candidate_counts[n // 2] if n > 0 else 0
    p95 = candidate_counts[int(n * 0.95)] if n > 0 else 0

    # Memory stats
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Database size
    db_size_bytes = get_db_size(DB_PATH)

    # Peak RSS
    peak_rss_mb = get_peak_rss_mb()

    # Print results
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"Sampled S1 entities     : {len(s1_entities):,}")
    print(f"Matched S1 entities     : {matched_entities:,}")
    print(f"True match links        : {total_true_links:,}")
    print(f"Covered true links      : {covered_links:,}")
    print(f"LINK RECALL             : {covered_links / total_true_links * 100:.2f}%")
    print()
    print(f"Entities with >=1 hit   : {covered_entities:,}")
    print(f"S1 ENTITY COVERAGE      : {covered_entities / matched_entities * 100:.2f}%")
    print()
    print(f"Average candidates/S1   : {total_candidates / processed:.1f}" if processed else "N/A")
    print(f"Median candidates/S1    : {p50}")
    print(f"P95 candidates/S1       : {p95}")
    print(f"Maximum candidates/S1   : {max_candidates}")
    print()
    print(f"Blocking index build    : {block_time:.1f}s")
    print(f"Candidate generation    : {eval_time:.1f}s")
    print(f"Total runtime           : {total_time:.1f}s ({total_time/60:.1f} min)")
    print()
    print(f"SQLite database size    : {db_size_bytes / (1024*1024):.1f} MB")
    print(f"Approx peak RAM (RSS)   : {peak_rss_mb:.1f} MB")
    print(f"Traced peak memory      : {peak_mem / (1024*1024):.1f} MB")
    print("=" * 60)

    return {
        "sampled_s1": len(s1_entities),
        "matched_entities": matched_entities,
        "total_links": total_true_links,
        "covered_links": covered_links,
        "link_recall_pct": covered_links / total_true_links * 100 if total_true_links > 0 else 0,
        "covered_entities": covered_entities,
        "entity_coverage_pct": covered_entities / matched_entities * 100 if matched_entities > 0 else 0,
        "avg_candidates": total_candidates / processed if processed else 0,
        "median_candidates": p50,
        "p95_candidates": p95,
        "max_candidates": max_candidates,
        "runtime_seconds": total_time,
        "db_size_mb": db_size_bytes / (1024 * 1024),
        "peak_rss_mb": peak_rss_mb
    }


if __name__ == "__main__":
    results = run_smoke_test()