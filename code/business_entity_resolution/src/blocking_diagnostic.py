import csv
import os
import re
import sqlite3
import time

ROOT = "/Users/kumar/Desktop/amazon_ml_entity_resolution"
TRAIN = os.path.join(ROOT, "dataset", "train")
DB = os.path.join(ROOT, "output", "train_blocking.sqlite")
DEFAULT_DB_PATH = DB

os.makedirs(os.path.dirname(DB), exist_ok=True)


def norm_text(value):
    value = (value or "").lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def build_db():
    if os.path.exists(DB):
        os.remove(DB)

    con = sqlite3.connect(DB)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE target (
            entity_id TEXT PRIMARY KEY,
            name_norm TEXT,
            addr_norm TEXT,
            country_norm TEXT
        )
    """)

    cur.execute("CREATE INDEX idx_name ON target(name_norm)")
    cur.execute("CREATE INDEX idx_addr ON target(addr_norm)")
    cur.execute("CREATE INDEX idx_country_name ON target(country_norm, name_norm)")
    cur.execute("CREATE INDEX idx_country_addr ON target(country_norm, addr_norm)")

    total = 0
    batch = []

    for filename in ("train_source2.tsv", "train_source3.tsv"):
        path = os.path.join(TRAIN, filename)
        print(f"\nIndexing {filename}")

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                batch.append((
                    row["entity_id"],
                    norm_text(row.get("business_name")),
                    norm_text(row.get("business_address")),
                    norm_text(row.get("country"))
                ))

                if len(batch) >= 10000:
                    cur.executemany(
                        "INSERT INTO target VALUES (?, ?, ?, ?)",
                        batch
                    )
                    con.commit()
                    total += len(batch)
                    batch.clear()

                    if total % 500000 == 0:
                        print(f"Indexed: {total:,}")

    if batch:
        cur.executemany(
            "INSERT INTO target VALUES (?, ?, ?, ?)",
            batch
        )
        con.commit()
        total += len(batch)

    print(f"\nTotal target records indexed: {total:,}")

    con.close()


def load_s1(sample_size=None):
    result = {}

    path = os.path.join(TRAIN, "train_source1.tsv")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if sample_size is not None and i >= sample_size:
                break
            result[row["entity_id"]] = (
                row.get("business_name", ""),
                row.get("business_address", ""),
                row.get("country", "")
            )

    return result


def candidates(cur, name, address, country):
    name = norm_text(name)
    address = norm_text(address)
    country = norm_text(country)

    ids = set()

    if name:
        rows = cur.execute(
            """
            SELECT entity_id
            FROM target
            WHERE country_norm = ?
              AND name_norm = ?
            """,
            (country, name)
        )
        ids.update(x[0] for x in rows)

    if address:
        rows = cur.execute(
            """
            SELECT entity_id
            FROM target
            WHERE country_norm = ?
              AND addr_norm = ?
            """,
            (country, address)
        )
        ids.update(x[0] for x in rows)

    if name:
        rows = cur.execute(
            """
            SELECT entity_id
            FROM target
            WHERE name_norm = ?
            LIMIT 100
            """,
            (name,)
        )
        ids.update(x[0] for x in rows)

    if address:
        rows = cur.execute(
            """
            SELECT entity_id
            FROM target
            WHERE addr_norm = ?
            LIMIT 100
            """,
            (address,)
        )
        ids.update(x[0] for x in rows)

    return ids


def evaluate(sample_size=None, db_path=DEFAULT_DB_PATH):
    print(f"\nLoading training Source 1{' (sample: ' + str(sample_size) + ')' if sample_size else ''}...")
    s1 = load_s1(sample_size)
    print(f"S1 loaded: {len(s1):,}")

    # Use the new SQLiteBlocker for candidate generation
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'code'))
    from business_entity_resolution.src.blocking import SQLiteBlocker

    target_tsv_paths = [
        os.path.join(TRAIN, "train_source2.tsv"),
        os.path.join(TRAIN, "train_source3.tsv")
    ]

    # Check if database exists, build if needed
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found, building...")
        blocker = SQLiteBlocker(target_tsv_paths, db_path=db_path)
    else:
        print(f"Using existing database: {db_path}")
        # We still need to create the blocker to access rare tokens etc.
        # But we can skip rebuilding the DB
        # For simplicity, just load and use it
        blocker = SQLiteBlocker.__new__(SQLiteBlocker)
        blocker.db_path = db_path
        blocker.name_prefix_len = 3
        blocker.addr_prefix_len = 3
        blocker.rare_token_min_freq = 2
        blocker.rare_token_max_freq = 100
        blocker.prefix_candidates_limit = 5000

        # Load rare tokens from database
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        # Get name tokens
        cur.execute("SELECT DISTINCT token FROM name_token_inv")
        blocker.rare_name_tokens = {row[0] for row in cur.fetchall()}
        # Get address tokens
        cur.execute("SELECT DISTINCT token FROM addr_token_inv")
        blocker.rare_addr_tokens = {row[0] for row in cur.fetchall()}
        con.close()

    gt_path = os.path.join(TRAIN, "train_ground_truth.tsv")

    total_links = 0
    covered_links = 0

    matched_entities = 0
    covered_entities = 0

    total_candidates = 0
    max_candidates = 0
    candidate_counts = []

    with open(gt_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader, 1):
            value = (row.get("matched_entity_ids") or "").strip()

            if not value:
                continue

            matched_entities += 1

            true_ids = {
                x.strip()
                for x in value.split(",")
                if x.strip()
            }

            total_links += len(true_ids)

            s1_id = row["source1_entity_id"]
            if s1_id not in s1:
                continue

            name, address, country = s1[s1_id]

            # Create row dict for blocker
            row_dict = {
                "entity_id": s1_id,
                "business_name": name,
                "business_address": address,
                "country": country
            }

            cand = blocker.candidates(row_dict)

            total_candidates += len(cand)
            candidate_counts.append(len(cand))
            if len(cand) > max_candidates:
                max_candidates = len(cand)

            hit = true_ids.intersection(cand)

            covered_links += len(hit)

            if hit:
                covered_entities += 1

            if i % 5000 == 0:
                print(
                    f"Processed {i:,} | "
                    f"link recall so far: "
                    f"{covered_links / total_links:.4%} | "
                    f"avg candidates: {total_candidates / i:.1f} | "
                    f"max candidates: {max_candidates}"
                )

    # Calculate percentiles
    candidate_counts.sort()
    n = len(candidate_counts)
    p50 = candidate_counts[n // 2] if n > 0 else 0
    p95 = candidate_counts[int(n * 0.95)] if n > 0 else 0
    p99 = candidate_counts[int(n * 0.99)] if n > 0 else 0

    print("\n========================================")
    print("BLOCKING DIAGNOSTIC")
    print("========================================")
    print(f"Matched S1 entities     : {matched_entities:,}")
    print(f"True match links        : {total_links:,}")
    print(f"Covered links           : {covered_links:,}")
    print(
        f"LINK RECALL             : "
        f"{covered_links / total_links:.4%}"
    )
    print()
    print(f"Entities with hit       : {covered_entities:,}")
    print(
        f"ENTITY COVERAGE         : "
        f"{covered_entities / matched_entities:.4%}"
    )
    print()
    print(f"Average candidates/S1   : {total_candidates / matched_entities:.1f}" if matched_entities else "N/A")
    print(f"Median candidates/S1    : {p50}")
    print(f"P95 candidates/S1       : {p95}")
    print(f"P99 candidates/S1       : {p99}")
    print(f"Max candidates/S1       : {max_candidates}")
    print("========================================")

    return {
        "matched_entities": matched_entities,
        "total_links": total_links,
        "covered_links": covered_links,
        "link_recall": covered_links / total_links if total_links > 0 else 0,
        "covered_entities": covered_entities,
        "entity_coverage": covered_entities / matched_entities if matched_entities > 0 else 0,
        "avg_candidates": total_candidates / matched_entities if matched_entities > 0 else 0,
        "p50_candidates": p50,
        "p95_candidates": p95,
        "p99_candidates": p99,
        "max_candidates": max_candidates
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run blocking diagnostic')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of S1 records to sample (default: full dataset)')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH,
                        help='Path to SQLite database')
    args = parser.parse_args()

    start = time.time()

    evaluate(args.sample, args.db_path)

    print(
        f"\nTotal time: "
        f"{(time.time() - start) / 60:.1f} minutes"
    )
