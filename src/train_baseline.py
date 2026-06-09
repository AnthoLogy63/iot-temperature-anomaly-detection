import numpy as np
from sklearn.ensemble import IsolationForest


def train_baseline(X, random_state=42):
    model = IsolationForest(
        contamination='auto',
        random_state=random_state,
        n_estimators=100
    )
    model.fit(X)
    return model


def predict_baseline(model, X):
    preds = model.predict(X)
    scores = model.decision_function(X)
    anomalies = np.where(preds == -1)[0]
    return anomalies, scores, preds
