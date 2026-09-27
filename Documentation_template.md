# Business Entity Resolution — Methodology

## 1. Methodology Used

We formulate entity resolution as a pairwise binary classification problem.
Each Source-1 record is compared with plausible Source-2/Source-3 candidates.
The final decision threshold is selected on an entity-level validation split
using the challenge macro F0.5 metric.

All transformations use only the supplied challenge data.

## 2. Candidate Generation / Blocking Strategy

Candidate generation combines multiple high-recall mechanisms:

- normalized exact business-name block
- normalized exact address block
- token inverted indexes for names and addresses
- country-aware candidate retrieval
- character n-gram TF-IDF nearest neighbors for names
- character n-gram TF-IDF nearest neighbors for addresses

The union of these blocks is passed to the matching model. The final
candidate file records the exact set scored by the model.

## 3. Model Architecture and Feature Engineering

A gradient-boosted binary classifier is trained on candidate pairs.

Features include:

- country equality
- normalized name exact match
- character-level name similarity
- token-sort and token-set name similarity
- name token Jaccard and containment
- normalized address similarity
- token-sort and token-set address similarity
- address token Jaccard and containment
- numeric-token overlap
- length differences
- name-prefix agreement
- address number overlap

The classifier uses only pairwise information available in the challenge
files.

## 4. Validation and Thresholding

The training entities are split by Source-1 entity, ensuring that the same
Source-1 entity cannot appear in both train and validation. Candidate-pair
scores are converted to entity-level predictions and the decision threshold
is tuned for macro F0.5.

This is precision-conscious because the challenge metric penalizes false
merges more strongly than missed links.

## 5. Reproducibility

The code contains the complete preprocessing, blocking, feature generation,
training and inference pipeline. Dependencies are pinned by major/minor
ranges in requirements.txt.

## 6. Compliance

No external entity-resolution API, government registry lookup, geocoding
service, or external business-data augmentation is used.
