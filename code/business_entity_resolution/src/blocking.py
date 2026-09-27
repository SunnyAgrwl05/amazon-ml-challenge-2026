"""
Enhanced disk-backed multi-signal blocker for entity resolution.

Implements high-recall candidate generation using SQLite with:
- Exact normalized name blocks
- Exact normalized address blocks
- Country + normalized name blocks
- Country + normalized address blocks
- Name prefix blocks (3-4 chars)
- Address prefix blocks (3-4 chars)
- Rare name token inverted index (frequency 2-100)
- Rare address token inverted index (frequency 2-100)
- House number + country blocks
- Postal fragment + country blocks

All candidate sets are UNIONed to maximize recall while controlling explosion
via frequency pruning and per-signal caps.

Country constraint: S1 entities only match S2/S3 candidates from the same country.
"""

import csv
import sqlite3
import re
import os
from collections import defaultdict
from typing import Set, List, Tuple, Dict, Any, Union
import pandas as pd
from .text_utils import norm_name, norm_address, norm_text


class SQLiteBlocker:
    """
    Disk-backed blocker using SQLite for memory-efficient candidate generation.
    """

    def __init__(self,
                 targets: Union[List[str], pd.DataFrame],
                 db_path: str = "output/train_blocking.sqlite",
                 name_prefix_len: int = 3,
                 addr_prefix_len: int = 3,
                 rare_token_min_freq: int = 2,
                 rare_token_max_freq: int = 100,
                 prefix_candidates_limit: int = 5000,
                 name_topk: int = 30,
                 address_topk: int = 30):
        """
        Initialize blocker by building SQLite index from target TSV files or DataFrame.

        Args:
            targets: List of paths to TSV files (source2, source3) OR a pandas DataFrame
            db_path: Path to SQLite database file
            name_prefix_len: Length of name prefix for blocking
            addr_prefix_len: Length of address prefix for blocking
            rare_token_min_freq: Minimum frequency for token to be considered rare
            rare_token_max_freq: Maximum frequency for token to be considered rare
            prefix_candidates_limit: Max candidates returned for prefix searches
            name_topk: (Unused, kept for compatibility)
            address_topk: (Unused, kept for compatibility)
        """
        self.db_path = db_path
        self.name_prefix_len = name_prefix_len
        self.addr_prefix_len = addr_prefix_len
        self.rare_token_min_freq = rare_token_min_freq
        self.rare_token_max_freq = rare_token_max_freq
        self.prefix_candidates_limit = prefix_candidates_limit

        # Ensure output directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Build the database
        self._build_database(targets)

        # Compute rare tokens for inverted indexes
        self._compute_rare_tokens()

        # Build inverted indexes for rare tokens
        self._build_rare_token_indexes()

    def _build_database(self, targets: Union[List[str], pd.DataFrame]):
        """Build SQLite database with target records and basic indexes."""
        print(f"Building SQLite database at {self.db_path}")

        # Remove existing database
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        # Connect and configure for fast inserts
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("PRAGMA cache_size=-500000")  # 500MB cache

        # Create table
        con.execute("""
            CREATE TABLE targets (
                rowid INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL,
                business_name TEXT,
                business_address TEXT,
                country TEXT,
                norm_name TEXT,
                norm_address TEXT,
                country_norm TEXT,
                name_prefix TEXT,
                addr_prefix TEXT,
                name_housenum TEXT,
                addr_postal TEXT
            )
        """)

        # Prepare insert statement
        insert_sql = """
            INSERT INTO targets
            (entity_id, business_name, business_address, country,
             norm_name, norm_address, country_norm,
             name_prefix, addr_prefix, name_housenum, addr_postal)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """

        total = 0
        batch_size = 10000

        if isinstance(targets, pd.DataFrame):
            # Handle DataFrame input
            print(f"  Indexing {len(targets):,} records from DataFrame")
            batch = []

            for _, row in targets.iterrows():
                entity_id = row.get("entity_id", "").strip()
                business_name = row.get("business_name", "") or ""
                business_address = row.get("business_address", "") or ""
                country = row.get("country", "") or ""

                # Normalizations
                norm_name_val = norm_name(business_name)
                norm_address_val = norm_address(business_address)
                country_norm_val = norm_text(country)

                # Prefixes
                name_prefix_val = norm_name_val[:self.name_prefix_len] if norm_name_val else ""
                addr_prefix_val = norm_address_val[:self.addr_prefix_len] if norm_address_val else ""

                # Extract house number from address (simple regex)
                housenum_val = ""
                if business_address:
                    # Look for digits at start of address or after space/#/
                    match = re.search(r'(?:^|\s|#)(\d+[a-zA-Z]?)', business_address)
                    if match:
                        housenum_val = match.group(1)

                # Extract postal-like fragment (5 digits, or letter-digit combos)
                postal_val = ""
                if business_address:
                    # Look for 5-digit zip, or letter-digit combos like A1A 1A1
                    match = re.search(r'\b\d{5}\b|[A-Z]\d[A-Z] ?\d[A-Z]\d', business_address.upper())
                    if match:
                        postal_val = match.group(0).replace(" ", "")
                    # Fallback: any 5-digit sequence
                    elif re.search(r'\d{5}', business_address):
                        postal_val = re.search(r'\d{5}', business_address).group(0)

                batch.append((
                    entity_id, business_name, business_address, country,
                    norm_name_val, norm_address_val, country_norm_val,
                    name_prefix_val, addr_prefix_val, housenum_val, postal_val
                ))

                if len(batch) >= batch_size:
                    con.executemany(insert_sql, batch)
                    con.commit()
                    total += len(batch)
                    batch.clear()

                    if total % 500000 == 0:
                        print(f"    Indexed: {total:,} records")

            # Insert remaining batch
            if batch:
                con.executemany(insert_sql, batch)
                con.commit()
                total += len(batch)

        else:
            # Handle list of file paths (original behavior)
            tsv_paths = targets
            for tsv_path in tsv_paths:
                print(f"  Indexing {os.path.basename(tsv_path)}")
                with open(tsv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    batch = []

                    for row in reader:
                        entity_id = row.get("entity_id", "").strip()
                        business_name = row.get("business_name", "") or ""
                        business_address = row.get("business_address", "") or ""
                        country = row.get("country", "") or ""

                        # Normalizations
                        norm_name_val = norm_name(business_name)
                        norm_address_val = norm_address(business_address)
                        country_norm_val = norm_text(country)

                        # Prefixes
                        name_prefix_val = norm_name_val[:self.name_prefix_len] if norm_name_val else ""
                        addr_prefix_val = norm_address_val[:self.addr_prefix_len] if norm_address_val else ""

                        # Extract house number from address (simple regex)
                        housenum_val = ""
                        if business_address:
                            # Look for digits at start of address or after space/#/
                            match = re.search(r'(?:^|\s|#)(\d+[a-zA-Z]?)', business_address)
                            if match:
                                housenum_val = match.group(1)

                        # Extract postal-like fragment (5 digits, or letter-digit combos)
                        postal_val = ""
                        if business_address:
                            # Look for 5-digit zip, or letter-digit combos like A1A 1A1
                            match = re.search(r'\b\d{5}\b|[A-Z]\d[A-Z] ?\d[A-Z]\d', business_address.upper())
                            if match:
                                postal_val = match.group(0).replace(" ", "")
                            # Fallback: any 5-digit sequence
                            elif re.search(r'\d{5}', business_address):
                                postal_val = re.search(r'\d{5}', business_address).group(0)

                        batch.append((
                            entity_id, business_name, business_address, country,
                            norm_name_val, norm_address_val, country_norm_val,
                            name_prefix_val, addr_prefix_val, housenum_val, postal_val
                        ))

                        if len(batch) >= batch_size:
                            con.executemany(insert_sql, batch)
                            con.commit()
                            total += len(batch)
                            batch.clear()

                            if total % 500000 == 0:
                                print(f"    Indexed: {total:,} records")

                    # Insert remaining batch
                    if batch:
                        con.executemany(insert_sql, batch)
                        con.commit()
                        total += len(batch)

        print(f"  Total target records indexed: {total:,}")

        # Create indexes for fast lookup
        print("  Creating database indexes...")
        con.execute("CREATE INDEX IF NOT EXISTS idx_name ON targets(norm_name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_addr ON targets(norm_address)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_country_name ON targets(country_norm, norm_name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_country_addr ON targets(country_norm, norm_address)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_name_prefix ON targets(name_prefix)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_addr_prefix ON targets(addr_prefix)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_country_housenum ON targets(country_norm, name_housenum)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_country_postal ON targets(country_norm, addr_postal)")
        con.commit()

        con.close()

    def _compute_rare_tokens(self):
        """Compute rare token frequencies for name and address fields using SQL-native aggregation."""
        print("  Computing token frequencies for rare token indexes...")

        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("PRAGMA cache_size=-500000")  # 500MB cache
        cur = con.cursor()

        # Create temporary tables for token counts using SQL GROUP BY (disk-backed aggregation)
        cur.execute("""
            CREATE TEMP TABLE name_token_counts AS
            SELECT token, COUNT(*) as freq
            FROM (
                SELECT norm_name as token FROM targets WHERE norm_name IS NOT NULL
                UNION ALL
                SELECT norm_address as token FROM targets WHERE norm_address IS NOT NULL
            )
            WHERE token IS NOT NULL AND token != ''
            GROUP BY token
        """)
        cur.execute("""
            CREATE TEMP TABLE addr_token_counts AS
            SELECT token, COUNT(*) as freq
            FROM (
                SELECT norm_name as token FROM targets WHERE norm_name IS NOT NULL
                UNION ALL
                SELECT norm_address as token FROM targets WHERE norm_address IS NOT NULL
            )
            WHERE token IS NOT NULL AND token != ''
            GROUP BY token
        """)

        # Filter to rare tokens (frequency between min and max)
        cur.execute("""
            SELECT token FROM name_token_counts
            WHERE freq >= ? AND freq <= ?
        """, (self.rare_token_min_freq, self.rare_token_max_freq))
        self.rare_name_tokens = {row[0] for row in cur.fetchall()}

        cur.execute("""
            SELECT token FROM addr_token_counts
            WHERE freq >= ? AND freq <= ?
        """, (self.rare_token_min_freq, self.rare_token_max_freq))
        self.rare_addr_tokens = {row[0] for row in cur.fetchall()}

        print(f"    Rare name tokens: {len(self.rare_name_tokens):,}")
        print(f"    Rare address tokens: {len(self.rare_addr_tokens):,}")

        # Drop temporary tables
        con.execute("DROP TABLE name_token_counts")
        con.execute("DROP TABLE addr_token_counts")
        con.close()

    def _build_rare_token_indexes(self):
        """Build inverted indexes for rare tokens."""
        print("  Building rare token inverted indexes...")

        con = sqlite3.connect(self.db_path)

        # Drop and recreate token tables
        con.execute("DROP TABLE IF EXISTS name_token_inv")
        con.execute("DROP TABLE IF EXISTS addr_token_inv")

        con.execute("""
            CREATE TABLE name_token_inv (
                token TEXT NOT NULL,
                target_rowid INTEGER NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE addr_token_inv (
                token TEXT NOT NULL,
                target_rowid INTEGER NOT NULL
            )
        """)

        # Populate indexes in batches
        insert_name_sql = "INSERT INTO name_token_inv (token, target_rowid) VALUES (?,?)"
        insert_addr_sql = "INSERT INTO addr_token_inv (token, target_rowid) VALUES (?,?)"

        cur = con.cursor()
        cur.execute("SELECT rowid, norm_name, norm_address FROM targets")

        batch_name = []
        batch_addr = []
        batch_size = 5000
        processed = 0

        for rowid, norm_name_val, norm_address_val in cur:
            if norm_name_val:
                for token in set(norm_name_val.split()):  # Dedupe within record
                    if token in self.rare_name_tokens:
                        batch_name.append((token, rowid))

            if norm_address_val:
                for token in set(norm_address_val.split()):  # Dedupe within record
                    if token in self.rare_addr_tokens:
                        batch_addr.append((token, rowid))

            if len(batch_name) >= batch_size:
                con.executemany(insert_name_sql, batch_name)
                batch_name.clear()
            if len(batch_addr) >= batch_size:
                con.executemany(insert_addr_sql, batch_addr)
                batch_addr.clear()

            processed += 1
            if processed % 500000 == 0:
                print(f"    Indexed tokens for {processed:,} records")

        # Insert remaining batches
        if batch_name:
            con.executemany(insert_name_sql, batch_name)
        if batch_addr:
            con.executemany(insert_addr_sql, batch_addr)

        con.commit()

        # Create indexes on token tables
        con.execute("CREATE INDEX IF NOT EXISTS idx_name_token ON name_token_inv(token)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_addr_token ON addr_token_inv(token)")
        con.commit()

        con.close()
        print(f"  Rare token indexes built")

    def candidates(self, row: Dict[str, str]) -> List[str]:
        """
        Generate candidate target entity IDs for a source record using UNION of all signals.

        Strict country constraint: only candidates from the same country as the source
        entity are returned. This dramatically reduces cross-country false positives,
        which is critical for F0.5 (precision-weighted) scoring.

        Args:
            row: Dictionary with keys 'entity_id', 'business_name', 'business_address', 'country'

        Returns:
            Sorted list of unique candidate entity IDs
        """
        # Extract and normalize fields
        entity_id = row.get("entity_id", "")
        business_name = row.get("business_name", "") or ""
        business_address = row.get("business_address", "") or ""
        country = row.get("country", "") or ""

        norm_name_val = norm_name(business_name)
        norm_address_val = norm_address(business_address)
        country_norm_val = norm_text(country)

        # Token sets for rare token lookup
        name_tokens = set(norm_name_val.split()) if norm_name_val else set()
        addr_tokens = set(norm_address_val.split()) if norm_address_val else set()

        # Start with empty candidate set
        candidate_ids = set()

        # Connect to database for queries
        con = sqlite3.connect(self.db_path)
        con.row_factory = lambda cursor, row: row[0]  # Return first column only
        cur = con.cursor()

        try:
            # 1. Exact normalized name block (country-constrained)
            if norm_name_val:
                if country_norm_val:
                    cur.execute("""
                        SELECT entity_id FROM targets
                        WHERE country_norm = ? AND norm_name = ?
                    """, (country_norm_val, norm_name_val))
                else:
                    cur.execute("""
                        SELECT entity_id FROM targets
                        WHERE norm_name = ?
                    """, (norm_name_val,))
                candidate_ids.update(cur.fetchall())

            # 2. Exact normalized address block (country-constrained)
            if norm_address_val:
                if country_norm_val:
                    cur.execute("""
                        SELECT entity_id FROM targets
                        WHERE country_norm = ? AND norm_address = ?
                    """, (country_norm_val, norm_address_val))
                else:
                    cur.execute("""
                        SELECT entity_id FROM targets
                        WHERE norm_address = ?
                    """, (norm_address_val,))
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

            # 5. Name prefix block (with limit, country-constrained)
            if norm_name_val and len(norm_name_val) >= self.name_prefix_len:
                name_prefix = norm_name_val[:self.name_prefix_len]
                if country_norm_val:
                    cur.execute("""
                        SELECT entity_id FROM targets
                        WHERE country_norm = ? AND name_prefix = ?
                        LIMIT ?
                    """, (country_norm_val, name_prefix, self.prefix_candidates_limit))
                else:
                    cur.execute("""
                        SELECT entity_id FROM targets
                        WHERE name_prefix = ?
                        LIMIT ?
                    """, (name_prefix, self.prefix_candidates_limit))
                candidate_ids.update(cur.fetchall())

            # 6. Address prefix block (with limit, country-constrained)
            if norm_address_val and len(norm_address_val) >= self.addr_prefix_len:
                addr_prefix = norm_address_val[:self.addr_prefix_len]
                if country_norm_val:
                    cur.execute("""
                        SELECT entity_id FROM targets
                        WHERE country_norm = ? AND addr_prefix = ?
                        LIMIT ?
                    """, (country_norm_val, addr_prefix, self.prefix_candidates_limit))
                else:
                    cur.execute("""
                        SELECT entity_id FROM targets
                        WHERE addr_prefix = ?
                        LIMIT ?
                    """, (addr_prefix, self.prefix_candidates_limit))
                candidate_ids.update(cur.fetchall())

            # 7. Rare name token blocks (country-constrained)
            for token in name_tokens & self.rare_name_tokens:
                if country_norm_val:
                    cur.execute("""
                        SELECT t.entity_id FROM targets t
                        JOIN name_token_inv n ON t.rowid = n.target_rowid
                        WHERE n.token = ? AND t.country_norm = ?
                    """, (token, country_norm_val))
                else:
                    cur.execute("""
                        SELECT t.entity_id FROM targets t
                        JOIN name_token_inv n ON t.rowid = n.target_rowid
                        WHERE n.token = ?
                    """, (token,))
                candidate_ids.update(cur.fetchall())

            # 8. Rare address token blocks (country-constrained)
            for token in addr_tokens & self.rare_addr_tokens:
                if country_norm_val:
                    cur.execute("""
                        SELECT t.entity_id FROM targets t
                        JOIN addr_token_inv a ON t.rowid = a.target_rowid
                        WHERE a.token = ? AND t.country_norm = ?
                    """, (token, country_norm_val))
                else:
                    cur.execute("""
                        SELECT t.entity_id FROM targets t
                        JOIN addr_token_inv a ON t.rowid = a.target_rowid
                        WHERE a.token = ?
                    """, (token,))
                candidate_ids.update(cur.fetchall())

            # 9. House number + country block
            # Extract house number again for consistency
            housenum_val = ""
            if business_address:
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


# Backward-compatible alias for existing code
Blocker = SQLiteBlocker