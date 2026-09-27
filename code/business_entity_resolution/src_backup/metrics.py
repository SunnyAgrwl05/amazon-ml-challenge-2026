import numpy as np

def f05(true_set, pred_set):
    if not true_set and not pred_set:
        return 1.0
    if not true_set:
        return 0.0 if pred_set else 1.0
    if not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    p = tp / len(pred_set)
    r = tp / len(true_set)
    if p == 0 and r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)

def macro_f05(truth, predictions):
    vals = [f05(truth.get(k, set()), predictions.get(k, set()))
            for k in truth]
    return float(np.mean(vals)) if vals else 0.0

def tune_threshold(scores_by_entity, truth, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.50, 0.991, 0.01)
    best_t, best_s = 0.82, -1
    for t in thresholds:
        pred = {}
        for sid, pairs in scores_by_entity.items():
            pred[sid] = {eid for eid, score in pairs if score >= t}
        s = macro_f05(truth, pred)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s
