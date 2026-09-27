#!/usr/bin/env python3
"""
Validator for Business Entity Resolution submission.

Checks:
1. File existence and format
2. Correct number of rows (matches test_source1 count)
3. All entity IDs are valid
4. Matching results are subset of candidates
5. Basic sanity checks
"""

import argparse
import csv
import sys
from pathlib import Path


def load_ids_from_tsv(filepath, id_column='entity_id'):
    """Load all IDs from a TSV file."""
    ids = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ids.add(row[id_column].strip())
    return ids


def parse_entity_ids(id_string):
    """Parse comma-separated entity IDs string into set."""
    if not id_string or id_string.strip() == '':
        return set()
    return {x.strip() for x in id_string.split(',') if x.strip()}


def validate_submission(matching_path, candidate_path, test_dir):
    """Validate submission files."""
    test_dir = Path(test_dir)

    print("Loading test source files...")
    # Load all valid entity IDs from test sources
    source1_ids = load_ids_from_tsv(test_dir / 'test_source1.tsv')
    source2_ids = load_ids_from_tsv(test_dir / 'test_source2.tsv')
    source3_ids = load_ids_from_tsv(test_dir / 'test_source3.tsv')
    valid_target_ids = source2_ids | source3_ids

    print(f"  Source1 entities: {len(source1_ids):,}")
    print(f"  Source2 entities: {len(source2_ids):,}")
    print(f"  Source3 entities: {len(source3_ids):,}")
    print(f"  Valid target IDs: {len(valid_target_ids):,}")

    # Check matching_results.tsv
    print("\nChecking matching_results.tsv...")
    matching_data = {}
    with open(matching_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != ['source1_entity_id', 'matched_entity_ids']:
            print(f"ERROR: Invalid header in matching_results.tsv: {reader.fieldnames}")
            return False

        for row_num, row in enumerate(reader, start=2):
            sid = row['source1_entity_id'].strip()
            matches = row['matched_entity_ids'].strip()

            # Check for duplicate source1_entity_id
            if sid in matching_data:
                print(f"ERROR: Duplicate source1_entity_id '{sid}' at line {row_num}")
                return False

            matching_data[sid] = matches

            # Validate entity ID format
            if not sid.startswith('S1-'):
                print(f"WARNING: Unexpected source1_entity_id format: {sid} at line {row_num}")

            # Validate matched entity IDs
            if matches:
                match_ids = parse_entity_ids(matches)
                # Check that all matched IDs are valid targets
                invalid_ids = match_ids - valid_target_ids
                if invalid_ids:
                    print(f"ERROR: Invalid matched entity IDs at line {row_num}: {invalid_ids}")
                    return False

    # Check that all source1 entities are present
    missing_s1 = source1_ids - set(matching_data.keys())
    extra_s1 = set(matching_data.keys()) - source1_ids
    if missing_s1:
        print(f"ERROR: Missing {len(missing_s1)} source1 entities in matching_results.tsv")
        if len(missing_s1) <= 5:
            print(f"  Missing: {missing_s1}")
        return False
    if extra_s1:
        print(f"ERROR: Extra {len(extra_s1)} source1 entities in matching_results.tsv")
        if len(extra_s1) <= 5:
            print(f"  Extra: {extra_s1}")
        return False

    print(f"  ✓ Matching results: {len(matching_data):,} rows (matches source1 count)")

    # Check candidate_pairs.tsv
    print("\nChecking candidate_pairs.tsv...")
    candidate_data = {}
    with open(candidate_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != ['source1_entity_id', 'candidate_entity_ids']:
            print(f"ERROR: Invalid header in candidate_pairs.tsv: {reader.fieldnames}")
            return False

        for row_num, row in enumerate(reader, start=2):
            sid = row['source1_entity_id'].strip()
            candidates = row['candidate_entity_ids'].strip()

            # Check for duplicate source1_entity_id
            if sid in candidate_data:
                print(f"ERROR: Duplicate source1_entity_id '{sid}' at line {row_num}")
                return False

            candidate_data[sid] = candidates

            # Validate candidate entity IDs
            if candidates:
                cand_ids = parse_entity_ids(candidates)
                # Check that all candidate IDs are valid targets
                invalid_ids = cand_ids - valid_target_ids
                if invalid_ids:
                    print(f"ERROR: Invalid candidate entity IDs at line {row_num}: {invalid_ids}")
                    return False

    # Check that all source1 entities are present
    missing_s1_cand = source1_ids - set(candidate_data.keys())
    extra_s1_cand = set(candidate_data.keys()) - source1_ids
    if missing_s1_cand:
        print(f"ERROR: Missing {len(missing_s1_cand)} source1 entities in candidate_pairs.tsv")
        if len(missing_s1_cand) <= 5:
            print(f"  Missing: {missing_s1_cand}")
        return False
    if extra_s1_cand:
        print(f"ERROR: Extra {len(extra_s1_cand)} source1 entities in candidate_pairs.tsv")
        if len(extra_s1_cand) <= 5:
            print(f"  Extra: {extra_s1_cand}")
        return False

    print(f"  ✓ Candidate pairs: {len(candidate_data):,} rows (matches source1 count)")

    # Check consistency: matching results should be subset of candidates
    print("\nChecking consistency between files...")
    errors = 0
    for sid in source1_ids:
        matches_str = matching_data.get(sid, '')
        candidates_str = candidate_data.get(sid, '')

        matches = parse_entity_ids(matches_str) if matches_str else set()
        candidates = parse_entity_ids(candidates_str) if candidates_str else set()

        # Matches should be subset of candidates
        if not matches.issubset(candidates):
            extra_matches = matches - candidates
            print(f"ERROR: Matched IDs not in candidates for {sid}: {extra_matches}")
            errors += 1
            if errors >= 5:
                print("  (Too many errors, stopping)")
                break

    if errors == 0:
        print("  ✓ All matches are valid subsets of candidates")
    else:
        print(f"  ✗ Found {errors} consistency errors")
        return False

    # Additional checks: country constraint if we can extract from data
    print("\nPerforming additional sanity checks...")

    # Check that we're not predicting matches for obviously wrong cases
    empty_matches = sum(1 for v in matching_data.values() if not v or v.strip() == '')
    print(f"  Entities with empty matches: {empty_matches:,} ({empty_matches/len(source1_ids)*100:.1f}%)")

    # Check candidate counts
    candidate_counts = []
    for sid, cand_str in candidate_data.items():
        if cand_str:
            count = len(parse_entity_ids(cand_str))
        else:
            count = 0
        candidate_counts.append(count)

    if candidate_counts:
        import statistics
        print(f"  Average candidates per entity: {statistics.mean(candidate_counts):.1f}")
        print(f"  Median candidates per entity: {statistics.median(candidate_counts):.1f}")
        print(f"  Max candidates per entity: {max(candidate_counts)}")

    print("\n" + "="*60)
    print("VALIDATION PASSED")
    print("="*60)
    print("All format and consistency checks passed.")
    print("Note: This validator does not check F0.5 score as ground truth is hidden.")
    print("To check F0.5, you would need access to the test labels.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate entity resolution submission')
    parser.add_argument('--matching', required=True, help='Path to matching_results.tsv')
    parser.add_argument('--candidate', required=True, help='Path to candidate_pairs.tsv')
    parser.add_argument('--test-dir', required=True, help='Path to test dataset directory')

    args = parser.parse_args()

    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()