# Amazon ML Challenge 2026 — Business Entity Resolution

## Approach

This pipeline is designed for the supplied challenge data only and does not use
external business databases, APIs, geocoding, or internet lookup.

### Pipeline

1. Normalize business names and addresses.
2. Generate high-recall candidates using:
   - exact normalized name/address
   - token inverted indexes
   - country-aware blocks
   - character TF-IDF nearest neighbors on name
   - character TF-IDF nearest neighbors on address
3. Build pairwise features:
   - exact matches
   - RapidFuzz character/token similarities
   - token Jaccard/containment
   - numeric-address overlap
   - country agreement
   - length/prefix features
4. Train an XGBoost binary pair classifier.
5. Split validation by Source-1 entity (not by pair) to avoid leakage.
6. Tune the final decision threshold directly on macro F0.5.
7. Infer test matches and output the required TSV files.
8. Candidate output is the exact candidate set scored by the matcher.

## Expected data layout

dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv

## Install

Python 3.10+ recommended.

pip install -r requirements.txt

## Train

python -m src.train --base .

## Infer

python -m src.infer --base .

Outputs:
- output/matching_results.tsv
- output/candidate_pairs.tsv

## Important

The challenge requires every Source-1 test entity to have exactly one output
row. Empty `matched_entity_ids` is used for predicted singletons.

Before submission, run the official validator supplied with the challenge:

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

The final threshold should be selected using a validation split of the
training entities and should not be tuned against the hidden test labels.
