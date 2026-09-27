import re
import unicodedata
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz


LEGAL = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "plc", "pvt", "private", "priv",
    "llp", "ltee"
}

ADDRESS_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "apt": "apartment",
    "fl": "floor", "no": "number"
}

@lru_cache(maxsize=300000)
def norm_text(x):
    x = "" if x is None else str(x)
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    x = x.lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

@lru_cache(maxsize=300000)
def norm_name(x):
    s = norm_text(x)
    toks = [t for t in s.split() if t not in LEGAL]
    return " ".join(toks)

@lru_cache(maxsize=300000)
def norm_address(x):
    s = norm_text(x)
    toks = [ADDRESS_ABBR.get(t, t) for t in s.split()]
    return " ".join(toks)

def tokens(s):
    return set(norm_text(s).split())

def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def containment(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))

def numeric_tokens(s):
    return set(re.findall(r"\d+", norm_text(s)))

def sim_bundle(a, b, field):
    if field == "name":
        x, y = norm_name(a), norm_name(b)
    else:
        x, y = norm_address(a), norm_address(b)

    if not x or not y:
        return (0.0, 0.0, 0.0, 0.0)

    return (
        fuzz.ratio(x, y) / 100.0,
        fuzz.token_sort_ratio(x, y) / 100.0,
        fuzz.token_set_ratio(x, y) / 100.0,
        jaccard(x.split(), y.split()),
    )
