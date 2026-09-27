from pathlib import Path
import csv
import sqlite3
import argparse
import re

from .text_utils import norm_name, norm_address, norm_text


def build_db(test_dir, db_path):
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-500000")

    con.execute("""
        CREATE TABLE targets (
            rowid INTEGER PRIMARY KEY,
            entity_id TEXT NOT NULL,
            business_name TEXT,
            business_address TEXT,
            country TEXT,
            norm_name TEXT,
            norm_address TEXT
        )
    """)

    files = [
        test_dir / "test_source2.tsv",
        test_dir / "test_source3.tsv"
    ]

    insert_sql = """
        INSERT INTO targets
        (entity_id,business_name,business_address,country,norm_name,norm_address)
        VALUES (?,?,?,?,?,?)
    """

    total = 0

    for fp in files:
        print(f"Indexing {fp.name} ...", flush=True)

        with open(fp, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            batch = []

            for r in reader:
                name = r.get("business_name", "") or ""
                addr = r.get("business_address", "") or ""

                batch.append((
                    r.get("entity_id", ""),
                    name,
                    addr,
                    r.get("country", "") or "",
                    norm_name(name),
                    norm_address(addr)
                ))

                if len(batch) >= 10000:
                    con.executemany(insert_sql, batch)
                    total += len(batch)
                    batch.clear()

                    if total % 500000 == 0:
                        print(f"  {total:,} targets indexed", flush=True)

            if batch:
                con.executemany(insert_sql, batch)
                total += len(batch)

    print(f"Total targets: {total:,}", flush=True)

    print("Creating indexes...", flush=True)

    con.execute("CREATE INDEX idx_name ON targets(norm_name)")
    con.execute("CREATE INDEX idx_addr ON targets(norm_address)")
    con.execute("CREATE INDEX idx_name_addr ON targets(norm_name,norm_address)")
    con.commit()

    print("Database ready.", flush=True)
    return con


def generate(base):
    base = Path(base)
    test_dir = base / "dataset" / "test"
    out = base / "output"
    out.mkdir(parents=True, exist_ok=True)

    db_path = out / "fast_targets.sqlite"

    con = build_db(test_dir, db_path)

    cur = con.cursor()

    s1_file = test_dir / "test_source1.tsv"
    matching_file = out / "matching_results.tsv"
    candidate_file = out / "candidate_pairs.tsv"

    with open(s1_file, "r", encoding="utf-8", errors="replace", newline="") as f, \
         open(matching_file, "w", encoding="utf-8", newline="") as mf, \
         open(candidate_file, "w", encoding="utf-8", newline="") as cf:

        reader = csv.DictReader(f, delimiter="\t")

        mw = csv.writer(mf, delimiter="\t")
        cw = csv.writer(cf, delimiter="\t")

        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])

        count = 0

        for r in reader:
            s1_id = r.get("entity_id", "")
            name = norm_name(r.get("business_name", "") or "")
            addr = norm_address(r.get("business_address", "") or "")

            ids = []

            # Strongest: exact normalized name + address
            if name and addr:
                cur.execute(
                    """
                    SELECT entity_id FROM targets
                    WHERE norm_name=? AND norm_address=?
                    """,
                    (name, addr)
                )
                ids = [x[0] for x in cur.fetchall()]

            # Fallback: exact normalized name
            if not ids and name:
                cur.execute(
                    "SELECT entity_id FROM targets WHERE norm_name=? LIMIT 20",
                    (name,)
                )
                ids = [x[0] for x in cur.fetchall()]

            # Fallback: exact normalized address
            if not ids and addr:
                cur.execute(
                    "SELECT entity_id FROM targets WHERE norm_address=? LIMIT 20",
                    (addr,)
                )
                ids = [x[0] for x in cur.fetchall()]

            # Remove duplicates while preserving order
            ids = list(dict.fromkeys(ids))

            cw.writerow([s1_id, ",".join(ids)])
            mw.writerow([s1_id, ",".join(ids)])

            count += 1

            if count % 100000 == 0:
                print(f"Processed Source-1: {count:,}", flush=True)

    con.close()

    print()
    print("DONE")
    print(f"Source-1 rows: {count:,}")
    print(f"Matching: {matching_file}")
    print(f"Candidates: {candidate_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="../..")
    args = parser.parse_args()
    generate(args.base)
