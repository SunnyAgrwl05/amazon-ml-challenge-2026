import pandas as pd

def load_sources(folder):
    s1 = pd.read_csv(folder / "source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(folder / "source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(folder / "source3.tsv", sep="\t", dtype=str).fillna("")
    return s1, s2, s3

def load_ground_truth(path):
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    truth = {}
    for _, r in gt.iterrows():
        ids = [x for x in str(r["matched_entity_ids"]).split(",") if x]
        truth[r["source1_entity_id"]] = set(ids)
    return truth
