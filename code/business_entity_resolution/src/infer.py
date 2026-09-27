from pathlib import Path
import argparse
import pandas as pd

from .blocking import Blocker
from .features import pair_features
from .model import Matcher
from .config import (
    NAME_TOPK, ADDRESS_TOPK, MAX_CANDIDATES_PER_SOURCE1,
    DEFAULT_THRESHOLD, MIN_STRING_SIMILARITY, COUNTRY_CONSTRAINT,
)
from .text_utils import norm_name, norm_address, norm_text


def _similarity(a, b):
    """Quick pre-filter similarity: name + address combined ratio."""
    an, bn = norm_name(a), norm_name(b)
    aa, ba = norm_address(a), norm_address(b)

    # If countries differ, similarity is 0 (strict country constraint)
    if COUNTRY_CONSTRAINT:
        if norm_text(a.country) != norm_text(b.country):
            return 0.0

    name_sim = 0.0
    if an and bn:
        from rapidfuzz import fuzz
        name_sim = fuzz.ratio(an, bn) / 100.0

    addr_sim = 0.0
    if aa and ba:
        from rapidfuzz import fuzz
        addr_sim = fuzz.ratio(aa, ba) / 100.0

    # Combined: max of name and address similarity
    return max(name_sim, addr_sim)


def main(base):
    base = Path(base)
    test = base / "dataset" / "test"
    out = base / "output"
    out.mkdir(exist_ok=True)

    s1 = pd.read_csv(test / "test_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(test / "test_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(test / "test_source3.tsv", sep="\t", dtype=str).fillna("")
    targets = pd.concat([s2, s3], ignore_index=True)

    blocker = Blocker(targets, NAME_TOPK, ADDRESS_TOPK)
    model = Matcher().load(out / "matcher.joblib")
    threshold = float((out / "threshold.txt").read_text().strip())

    match_rows = []
    candidate_rows = []

    for _, a in s1.iterrows():
        cand_idx = blocker.candidates(a)
        # Keep all exact/near candidates produced by the blocker. If unusually
        # large, retain the highest model scores rather than arbitrary rows.
        scored = []
        for j in cand_idx:
            b = targets.iloc[j]

            # Pre-filter: skip pairs with low string similarity
            sim = _similarity(a, b)
            if sim < MIN_STRING_SIMILARITY:
                continue

            p = float(model.predict_proba(pair_features(a, b).reshape(1, -1))[0])
            scored.append((b.entity_id, p))

        scored.sort(key=lambda x: x[1], reverse=True)
        if len(scored) > MAX_CANDIDATES_PER_SOURCE1:
            scored = scored[:MAX_CANDIDATES_PER_SOURCE1]

        candidates = [eid for eid, _ in scored]
        matches = [eid for eid, p in scored if p >= threshold]

        candidate_rows.append({
            "source1_entity_id": a.entity_id,
            "candidate_entity_ids": ",".join(candidates)
        })
        match_rows.append({
            "source1_entity_id": a.entity_id,
            "matched_entity_ids": ",".join(matches)
        })

    pd.DataFrame(match_rows).to_csv(out / "matching_results.tsv",
                                    sep="\t", index=False)
    pd.DataFrame(candidate_rows).to_csv(out / "candidate_pairs.tsv",
                                        sep="\t", index=False)
    print(f"Wrote {len(match_rows):,} Source-1 rows")
    print(f"Threshold: {threshold:.4f}")
    print(f"Min string similarity pre-filter: {MIN_STRING_SIMILARITY}")
    print(f"Country constraint: {COUNTRY_CONSTRAINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    args = parser.parse_args()
    main(args.base)