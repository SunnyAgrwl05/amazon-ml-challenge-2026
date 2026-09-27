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

# Precision-first inference. This is overwritten by validation tuning.
DEFAULT_THRESHOLD = 0.82

# Keep candidates reasonably small while preserving recall.
MAX_CANDIDATES_PER_SOURCE1 = 120
