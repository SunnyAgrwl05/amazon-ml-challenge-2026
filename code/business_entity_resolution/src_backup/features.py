import numpy as np
from rapidfuzz import fuzz

from .text_utils import norm_name, norm_address, norm_text, tokens, jaccard, containment, numeric_tokens

FEATURE_NAMES = [
    "country_eq", "name_ratio", "name_token_sort", "name_token_set",
    "name_jaccard", "name_containment", "addr_ratio", "addr_token_sort",
    "addr_token_set", "addr_jaccard", "addr_containment", "num_overlap",
    "name_len_diff", "addr_len_diff", "name_exact", "addr_exact",
    "name_prefix", "address_has_number_overlap"
]

def pair_features(a, b):
    an, bn = norm_name(a.business_name), norm_name(b.business_name)
    aa, ba = norm_address(a.business_address), norm_address(b.business_address)

    nt1, nt2 = an.split(), bn.split()
    at1, at2 = aa.split(), ba.split()
    nums1, nums2 = numeric_tokens(aa), numeric_tokens(ba)

    def safe_ratio(x, y):
        return fuzz.ratio(x, y) / 100.0 if x and y else 0.0

    return np.array([
        float(norm_text(a.country) == norm_text(b.country) and norm_text(a.country) != ""),
        safe_ratio(an, bn),
        fuzz.token_sort_ratio(an, bn) / 100.0 if an and bn else 0.0,
        fuzz.token_set_ratio(an, bn) / 100.0 if an and bn else 0.0,
        jaccard(nt1, nt2),
        containment(nt1, nt2),
        safe_ratio(aa, ba),
        fuzz.token_sort_ratio(aa, ba) / 100.0 if aa and ba else 0.0,
        fuzz.token_set_ratio(aa, ba) / 100.0 if aa and ba else 0.0,
        jaccard(at1, at2),
        containment(at1, at2),
        jaccard(nums1, nums2),
        abs(len(an) - len(bn)) / max(len(an), len(bn), 1),
        abs(len(aa) - len(ba)) / max(len(aa), len(ba), 1),
        float(an != "" and an == bn),
        float(aa != "" and aa == ba),
        float(an[:5] != "" and an[:5] == bn[:5]),
        float(bool(nums1 & nums2)),
    ], dtype=float)
