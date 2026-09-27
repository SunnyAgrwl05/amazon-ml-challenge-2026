#!/usr/bin/env python3
"""
Blocking validation framework.
Tests multiple blocking strategies on a deterministic ~8K S1 validation sample.
Reports: candidate count, link recall, entity coverage, zero-candidate S1, avg candidates/S1, p95 candidates/S1
"""

import csv
import os
import sys
import time
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Callable
from collections import defaultdict

ROOT = Path("/Users/kumar/Desktop/amazon_ml_entity_resolution")
TRAIN = ROOT / "dataset" / "train"
DB_PATH = ROOT / "output" / "train_blocking.sqlite"
SAMPLE_SIZE = 8000  # Deterministic validation sample
RANDOM_SEED = 42

sys.path.insert(0, str(ROOT / 'code'))
from business_entity_resolution.src.blocking import SQLiteBlocker
from business_entity_resolution.src.text_utils import norm_name, norm_address, norm_text


def load_s1_entities(sample_size: int) -> Dict[str, Tuple[str, str, str]]:
    """Load first `sample_size` entities from train_source1.tsv"""
    path = TRAIN / "train_source1.tsv"
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
    path = TRAIN / "train_ground_truth.tsv"
    result = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            if s1_id in s1_ids:
                value = (row.get("matched_entity_ids") or "").strip()
                if value:
                    true_ids = {x.strip() for x in value.split(",") if x.strip()}
                    result[s1_id] = true_ids
                else:
                    result[s1_id] = set()
    return result


class FixedSQLiteBlocker:
    """SQLiteBlocker with FIXED rare token computation."""

    def __init__(self, db_path: str, prefix_candidates_limit: int = 5000):
        self.db_path = db_path
        self.name_prefix_len = 3
        self.addr_prefix_len = 3
        self.rare_token_min_freq = 2
        self.rare_token_max_freq = 100
        self.prefix_candidates_limit = prefix_candidates_limit

        # Load rare tokens with CORRECT separation
        self._load_rare_tokens()

    def _load_rare_tokens(self):
        """Load rare tokens from database with correct separation."""
        import sqlite3
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

        # Name tokens from name_token_inv table
        cur.execute("SELECT DISTINCT token FROM name_token_inv")
        self.rare_name_tokens = {row[0] for row in cur.fetchall()}

        # Address tokens from addr_token_inv table
        cur.execute("SELECT DISTINCT token FROM addr_token_inv")
        self.rare_addr_tokens = {row[0] for row in cur.fetchall()}

        con.close()
        print(f"  Loaded rare name tokens: {len(self.rare_name_tokens):,}")
        print(f"  Loaded rare addr tokens: {len(self.rare_addr_tokens):,}")
        print(f"  Intersection: {len(self.rare_name_tokens & self.rare_addr_tokens):,}")

    def candidates(self, row: Dict[str, str]) -> List[str]:
        """Generate candidates using all 10 signals."""
        entity_id = row.get("entity_id", "")
        business_name = row.get("business_name", "") or ""
        business_address = row.get("business_address", "") or ""
        country = row.get("country", "") or ""

        norm_name_val = norm_name(business_name)
        norm_address_val = norm_address(business_address)
        country_norm_val = norm_text(country)

        name_tokens = set(norm_name_val.split()) if norm_name_val else set()
        addr_tokens = set(norm_address_val.split()) if norm_address_val else set()

        candidate_ids = set()

        con = sqlite3.connect(self.db_path)
        con.row_factory = lambda cursor, row: row[0]
        cur = con.cursor()

        try:
            # 1. Exact normalized name block
            if norm_name_val:
                cur.execute("SELECT entity_id FROM targets WHERE norm_name = ?", (norm_name_val,))
                candidate_ids.update(cur.fetchall())

            # 2. Exact normalized address block
            if norm_address_val:
                cur.execute("SELECT entity_id FROM targets WHERE norm_address = ?", (norm_address_val,))
                candidate_ids.update(cur.fetchall())

            # 3. Country + normalized name block
            if norm_name_val and country_norm_val:
                cur.execute("""
                    SELECT entity_id FROM targets
                    WHERE country_norm = ? AND norm_name = ?
                """, (country_norm_val, norm_name_val))
                candidate_ids.update(cur.fetchall())

            # 4. Country + normalized address block
            if norm_address_val and country_norm_val:
                cur.execute("""
                    SELECT entity_id FROM targets
                    WHERE country_norm = ? AND norm_address = ?
                """, (country_norm_val, norm_address_val))
                candidate_ids.update(cur.fetchall())

            # 5. Name prefix block (with limit)
            if norm_name_val and len(norm_name_val) >= self.name_prefix_len:
                name_prefix = norm_name_val[:self.name_prefix_len]
                cur.execute("""
                    SELECT entity_id FROM targets
                    WHERE name_prefix = ? LIMIT ?
                """, (name_prefix, self.prefix_candidates_limit))
                candidate_ids.update(cur.fetchall())

            # 6. Address prefix block (with limit)
            if norm_address_val and len(norm_address_val) >= self.addr_prefix_len:
                addr_prefix = norm_address_val[:self.addr_prefix_len]
                cur.execute("""
                    SELECT entity_id FROM targets
                    WHERE addr_prefix = ? LIMIT ?
                """, (addr_prefix, self.prefix_candidates_limit))
                candidate_ids.update(cur.fetchall())

            # 7. Rare name token blocks (FIXED: uses rare_name_tokens)
            for token in name_tokens & self.rare_name_tokens:
                cur.execute("""
                    SELECT t.entity_id FROM targets t
                    JOIN name_token_inv n ON t.rowid = n.target_rowid
                    WHERE n.token = ?
                """, (token,))
                candidate_ids.update(cur.fetchall())

            # 8. Rare address token blocks (FIXED: uses rare_addr_tokens)
            for token in addr_tokens & self.rare_addr_tokens:
                cur.execute("""
                    SELECT t.entity_id FROM targets t
                    JOIN addr_token_inv a ON t.rowid = a.target_rowid
                    WHERE a.token = ?
                """, (token,))
                candidate_ids.update(cur.fetchall())

            # 9. House number + country block
            housenum_val = ""
            if business_address:
                import re
                match = re.search(r'(?:^|\s|#)(\d+[a-zA-Z]?)', business_address)
                if match:
                    housenum_val = match.group(1)

            if housenum_val and country_norm_val:
                cur.execute("""
                    SELECT entity_id FROM targets
                    WHERE country_norm = ? AND name_housenum = ?
                """, (country_norm_val, housenum_val))
                candidate_ids.update(cur.fetchall())

            # 10. Postal fragment + country block
            postal_val = ""
            if business_address:
                import re
                match = re.search(r'\b\d{5}\b|[A-Z]\d[A-Z] ?\d[A-Z]\d', business_address.upper())
                if match:
                    postal_val = match.group(0).replace(" ", "")
                elif re.search(r'\d{5}', business_address):
                    postal_val = re.search(r'\d{5}', business_address).group(0)

            if postal_val and country_norm_val:
                cur.execute("""
                    SELECT entity_id FROM targets
                    WHERE country_norm = ? AND addr_postal = ?
                """, (country_norm_val, postal_val))
                candidate_ids.update(cur.fetchall())

        finally:
            con.close()

        return sorted(candidate_ids)


# Individual blocking signal functions for comparison

def signal_exact_name(row: Dict, con) -> Set[str]:
    """1. Exact normalized name"""
    norm_name_val = norm_name(row.get("business_name", "") or "")
    if not norm_name_val:
        return set()
    cur = con.cursor()
    cur.execute("SELECT entity_id FROM targets WHERE norm_name = ?", (norm_name_val,))
    return set(cur.fetchall())


def signal_exact_addr(row: Dict, con) -> Set[str]:
    """2. Exact normalized address"""
    norm_addr_val = norm_address(row.get("business_address", "") or "")
    if not norm_addr_val:
        return set()
    cur = con.cursor()
    cur.execute("SELECT entity_id FROM targets WHERE norm_address = ?", (norm_addr_val,))
    return set(cur.fetchall())


def signal_name_or_addr(row: Dict, con) -> Set[str]:
    """3. Normalized name OR address (union)"""
    ids = set()
    norm_name_val = norm_name(row.get("business_name", "") or "")
    norm_addr_val = norm_address(row.get("business_address", "") or "")
    if norm_name_val:
        cur = con.cursor()
        cur.execute("SELECT entity_id FROM targets WHERE norm_name = ?", (norm_name_val,))
        ids.update(cur.fetchall())
    if norm_addr_val:
        cur = con.cursor()
        cur.execute("SELECT entity_id FROM targets WHERE norm_address = ?", (norm_addr_val,))
        ids.update(cur.fetchall())
    return ids


def signal_name_postal_housenum(row: Dict, con) -> Set[str]:
    """4. Name + postal/house number combinations"""
    ids = set()
    business_name = row.get("business_name", "") or ""
    business_address = row.get("business_address", "") or ""
    country = row.get("country", "") or ""

    norm_name_val = norm_name(business_name)
    country_norm_val = norm_text(country)

    if not norm_name_val or not country_norm_val:
        return ids

    cur = con.cursor()

    # Name + country
    cur.execute("""
        SELECT entity_id FROM targets
        WHERE country_norm = ? AND norm_name = ?
    """, (country_norm_val, norm_name_val))
    ids.update(cur.fetchall())

    # Name + house number + country
    import re
    housenum_val = ""
    if business_address:
        match = re.search(r'(?:^|\s|#)(\d+[a-zA-Z]?)', business_address)
        if match:
            housenum_val = match.group(1)
    if housenum_val:
        cur.execute("""
            SELECT entity_id FROM targets
            WHERE country_norm = ? AND norm_name = ? AND name_housenum = ?
        """, (country_norm_val, norm_name_val, housenum_val))
        ids.update(cur.fetchall())

    # Name + postal + country
    postal_val = ""
    if business_address:
        match = re.search(r'\b\d{5}\b|[A-Z]\d[A-Z] ?\d[A-Z]\d', business_address.upper())
        if match:
            postal_val = match.group(0).replace(" ", "")
        elif re.search(r'\d{5}', business_address):
            postal_val = re.search(r'\d{5}', business_address).group(0)
    if postal_val:
        cur.execute("""
            SELECT entity_id FROM targets
            WHERE country_norm = ? AND norm_name = ? AND addr_postal = ?
        """, (country_norm_val, norm_name_val, postal_val))
        ids.update(cur.fetchall())

    return ids


def signal_multi_union(blocker: FixedSQLiteBlocker, row: Dict) -> Set[str]:
    """5. Full multi-signal union (using fixed blocker)"""
    return set(blocker.candidates(row))


def evaluate_blocking_signal(
    signal_name: str,
    signal_func: Callable,
    s1_entities: Dict[str, Tuple[str, str, str]],
    ground_truth: Dict[str, Set[str]],
    con,
    blocker: FixedSQLiteBlocker = None
) -> Dict:
    """Evaluate a single blocking signal."""

    total_true_links = 0
    covered_links = 0
    matched_entities = 0
    covered_entities = 0
    total_candidates = 0
    max_candidates = 0
    candidate_counts = []
    zero_candidate_entities = 0
    processed = 0

    for s1_id, (name, address, country) in s1_entities.items():
        if s1_id not in ground_truth:
            continue

        true_ids = ground_truth[s1_id]
        if not true_ids:
            continue

        matched_entities += 1
        total_true_links += len(true_ids)

        row_dict = {
            "entity_id": s1_id,
            "business_name": name,
            "business_address": address,
            "country": country
        }

        if blocker and signal_name == "multi_union":
            candidates = signal_func(blocker, row_dict)
        else:
            candidates = signal_func(row_dict, con)

        cand_count = len(candidates)
        total_candidates += cand_count
        candidate_counts.append(cand_count)
        max_candidates = max(max_candidates, cand_count)

        if cand_count == 0:
            zero_candidate_entities += 1

        hit = true_ids.intersection(candidates)
        covered_links += len(hit)

        if hit:
            covered_entities += 1

        processed += 1

    candidate_counts.sort()
    n = len(candidate_counts)
    p50 = candidate_counts[n // 2] if n > 0 else 0
    p95 = candidate_counts[int(n * 0.95)] if n > 0 else 0
    p99 = candidate_counts[int(n * 0.99)] if n > 0 else 0

    return {
        "signal": signal_name,
        "matched_entities": matched_entities,
        "total_true_links": total_true_links,
        "covered_links": covered_links,
        "link_recall_pct": covered_links / total_true_links * 100 if total_true_links > 0 else 0,
        "covered_entities": covered_entities,
        "entity_coverage_pct": covered_entities / matched_entities * 100 if matched_entities > 0 else 0,
        "zero_candidate_entities": zero_candidate_entities,
        "zero_candidate_pct": zero_candidate_entities / matched_entities * 100 if matched_entities > 0 else 0,
        "avg_candidates": total_candidates / processed if processed > 0 else 0,
        "median_candidates": p50,
        "p95_candidates": p95,
        "p99_candidates": p99,
        "max_candidates": max_candidates
    }


def print_results(results: Dict):
    print(f"\n{'='*70}")
    print(f"  {results['signal']}")
    print(f"{'='*70}")
    print(f"  Matched S1 entities       : {results['matched_entities']:,}")
    print(f"  True match links          : {results['total_true_links']:,}")
    print(f"  Covered true links        : {results['covered_links']:,}")
    print(f"  LINK RECALL               : {results['link_recall_pct']:.2f}%")
    print(f"")
    print(f"  Entities with >=1 hit     : {results['covered_entities']:,}")
    print(f"  ENTITY COVERAGE           : {results['entity_coverage_pct']:.2f}%")
    print(f"")
    print(f"  Zero-candidate entities   : {results['zero_candidate_entities']:,} ({results['zero_candidate_pct']:.2f}%)")
    print(f"  Average candidates/S1     : {results['avg_candidates']:.1f}")
    print(f"  Median candidates/S1      : {results['median_candidates']}")
    print(f"  P95 candidates/S1         : {results['p95_candidates']}")
    print(f"  P99 candidates/S1         : {results['p99_candidates']}")
    print(f"  Max candidates/S1         : {results['max_candidates']:,}")


def run_validation():
    print(f"{'='*70}")
    print(f"BLOCKING VALIDATION - {SAMPLE_SIZE:,} S1 ENTITIES (seed={RANDOM_SEED})")
    print(f"{'='*70}")
    print(f"Database: {DB_PATH}")
    print(f"Sample size: {SAMPLE_SIZE:,}")
    print()

    # Load validation sample
    print("Loading S1 entities...")
    s1_entities = load_s1_entities(SAMPLE_SIZE)
    s1_ids = set(s1_entities.keys())
    print(f"  Loaded: {len(s1_entities):,}")

    print("Loading ground truth...")
    ground_truth = load_ground_truth(s1_ids)
    matched_entities = len([k for k, v in ground_truth.items() if v])
    total_links = sum(len(v) for v in ground_truth.values())
    print(f"  Matched S1 entities: {matched_entities:,}")
    print(f"  True match links: {total_links:,}")
    print()

    # Load fixed blocker
    print("Loading fixed blocker...")
    blocker = FixedSQLiteBlocker(str(DB_PATH))
    print()

    # Connect to DB for signal functions
    import sqlite3
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = lambda cursor, row: row[0]

    try:
        # Define signals to test
        signals = [
            ("exact_normalized_name", signal_exact_name),
            ("exact_normalized_address", signal_exact_addr),
            ("name_OR_address", signal_name_or_addr),
            ("name_postal_housenum", signal_name_postal_housenum),
            ("multi_union", signal_multi_union),
        ]

        all_results = []

        for signal_name, signal_func in signals:
            print(f"Evaluating: {signal_name}...")
            start = time.time()

            if signal_name == "multi_union":
                results = evaluate_blocking_signal(
                    signal_name, signal_func, s1_entities, ground_truth, con, blocker
                )
            else:
                results = evaluate_blocking_signal(
                    signal_name, signal_func, s1_entities, ground_truth, con
                )

            elapsed = time.time() - start
            results["elapsed_sec"] = elapsed
            all_results.append(results)
            print_results(results)

        # Summary table
        print(f"\n{'='*70}")
        print(f"SUMMARY COMPARISON")
        print(f"{'='*70}")
        print(f"{'Signal':<30} {'Link Recall':>12} {'Entity Cov.':>12} {'Zero %':>8} {'Avg Cand':>10} {'P95':>8} {'Time(s)':>8}")
        print(f"{'-'*70}")
        for r in all_results:
            print(f"{r['signal']:<30} {r['link_recall_pct']:>11.2f}% {r['entity_coverage_pct']:>11.2f}% "
                  f"{r['zero_candidate_pct']:>7.2f}% {r['avg_candidates']:>10.1f} {r['p95_candidates']:>8} {r['elapsed_sec']:>8.1f}")

        return all_results

    finally:
        con.close()


if __name__ == "__main__":
    run_validation()