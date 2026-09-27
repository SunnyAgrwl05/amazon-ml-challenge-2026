from pathlib import Path
import joblib
import numpy as np
from xgboost import XGBClassifier

class Matcher:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=450,
            max_depth=6,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X, y):
        # Balanced weighting helps learn rare positives without forcing
        # a low inference threshold.
        pos = max(int(y.sum()), 1)
        neg = max(len(y) - pos, 1)
        self.model.set_params(scale_pos_weight=min(neg / pos, 25.0))
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
