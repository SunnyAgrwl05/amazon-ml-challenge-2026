from pathlib import Path
import csv
import sqlite3

from .text_utils import norm_name, norm_address


def main():
    base = Path("../..")
    test = base / "dataset" / "test"
    out = base / "output"

    db = out / "fast_targets.sqlite"
    matching = out / "matching_results.tsv"

    con = sqlite3.connect(db)
    cur = con.cursor()

    total = 0
    exact_both = 0
    unique_name = 0
    unique_addr = 0
    empty = 0

    with open(test / "test_source1.tsv", "r", encoding="utf-8",
              errors="replace", newline="") as f, \
         open(matching, "w", encoding="utf-8", newline="") as o:

        reader = csv.DictReader(f, delimiter="\t")
        writer = csv.writer(o, delimiter="\t")

        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for r in reader:
            s1 = r.get("entity_id", "")
            name = norm_name(r.get("business_name", "") or "")
            addr = norm_address(r.get("business_address", "") or "")

            ids = []

            # 1. Exact normalized name + address
            if name and addr:
                cur.execute(
                    "SELECT entity_id FROM targets "
                    "WHERE norm_name=? AND norm_address=?",
                    (name, addr)
                )
                ids = [x[0] for x in cur.fetchall()]

                if ids:
                    exact_both += 1

            # 2. If no exact pair, accept ONLY unique normalized name
            if not ids and name:
                cur.execute(
                    "SELECT entity_id FROM targets "
                    "WHERE norm_name=? LIMIT 2",
                    (name,)
                )
                rows = cur.fetchall()

                if len(rows) == 1:
                    ids = [rows[0][0]]
                    unique_name += 1

            # 3. If still no match, accept ONLY unique normalized address
            if not ids and addr:
                cur.execute(
                    "SELECT entity_id FROM targets "
                    "WHERE norm_address=? LIMIT 2",
                    (addr,)
                )
                rows = cur.fetchall()

                if len(rows) == 1:
                    ids = [rows[0][0]]
                    unique_addr += 1

            if not ids:
                empty += 1

            writer.writerow([s1, ",".join(dict.fromkeys(ids))])

            total += 1

            if total % 100000 == 0:
                print(
                    f"Processed {total:,} | "
                    f"exact={exact_both:,} "
                    f"unique_name={unique_name:,} "
                    f"unique_addr={unique_addr:,} "
                    f"empty={empty:,}",
                    flush=True
                )

    con.close()

    print()
    print("FINAL MATCHING COMPLETE")
    print(f"Rows: {total:,}")
    print(f"Exact name+address: {exact_both:,}")
    print(f"Unique name: {unique_name:,}")
    print(f"Unique address: {unique_addr:,}")
    print(f"Empty: {empty:,}")
    print(f"Output: {matching}")


if __name__ == "__main__":
    main()
