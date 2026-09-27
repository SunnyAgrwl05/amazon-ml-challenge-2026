from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
DATA = BASE / "dataset"
TRAIN = DATA / "train"
TEST = DATA / "test"
OUTPUT = BASE / "output"

# Candidate-generation limits
NAME_TOPK = 30
ADDRESS_TOPK = 30
COMBINED_TOPK = 50

# Precision-first inference. High threshold for F0.5 optimization.
DEFAULT_THRESHOLD = 0.88

# Minimum string similarity score before feeding to XGBoost
MIN_STRING_SIMILARITY = 0.70

# Keep candidates reasonably small while preserving recall.
MAX_CANDIDATES_PER_SOURCE1 = 120

# Strict country constraint: only match S1-S2/S3 from same country
COUNTRY_CONSTRAINT = True
