import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .text_utils import norm_name, norm_address, norm_text

class Blocker:
    """
    High-recall candidate generation:
      1) exact normalized-name / normalized-address blocks
      2) country-aware token blocks
      3) character TF-IDF nearest-neighbor retrieval on name
      4) character TF-IDF nearest-neighbor retrieval on address
    No external data is used.
    """

    def __init__(self, target_df, name_topk=30, address_topk=30):
        self.target = target_df.reset_index(drop=True).copy()
        self.name_topk = name_topk
        self.address_topk = address_topk

        self.name_inv = defaultdict(set)
        self.addr_inv = defaultdict(set)
        self.country_inv = defaultdict(set)

        for i, r in self.target.iterrows():
            n = norm_name(r.business_name)
            a = norm_address(r.business_address)
            c = norm_text(r.country)
            if n:
                self.name_inv[n].add(i)
                for t in set(n.split()):
                    if len(t) >= 3:
                        self.name_inv[f"tok:{t}"].add(i)
            if a:
                self.addr_inv[a].add(i)
                for t in set(a.split()):
                    if len(t) >= 4:
                        self.addr_inv[f"tok:{t}"].add(i)
            if c:
                self.country_inv[c].add(i)

        self.name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                        min_df=1, max_features=250000,
                                        sublinear_tf=True)
        self.addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                        min_df=1, max_features=250000,
                                        sublinear_tf=True)
        self.name_mat = self.name_vec.fit_transform(self.target.business_name.map(norm_name))
        self.addr_mat = self.addr_vec.fit_transform(self.target.business_address.map(norm_address))

        self.name_nn = NearestNeighbors(
            n_neighbors=min(name_topk, len(self.target)),
            metric="cosine", algorithm="brute"
        ).fit(self.name_mat)
        self.addr_nn = NearestNeighbors(
            n_neighbors=min(address_topk, len(self.target)),
            metric="cosine", algorithm="brute"
        ).fit(self.addr_mat)

    def candidates(self, row):
        ids = set()
        n = norm_name(row.business_name)
        a = norm_address(row.business_address)
        c = norm_text(row.country)

        ids.update(self.name_inv.get(n, set()))
        ids.update(self.addr_inv.get(a, set()))
        ids.update(self.country_inv.get(c, set()))

        for t in set(n.split()):
            if len(t) >= 3:
                ids.update(self.name_inv.get(f"tok:{t}", set()))
        for t in set(a.split()):
            if len(t) >= 4:
                ids.update(self.addr_inv.get(f"tok:{t}", set()))

        qn = self.name_vec.transform([n])
        _, ni = self.name_nn.kneighbors(qn)
        ids.update(ni[0].tolist())

        qa = self.addr_vec.transform([a])
        _, ai = self.addr_nn.kneighbors(qa)
        ids.update(ai[0].tolist())

        return sorted(ids)
