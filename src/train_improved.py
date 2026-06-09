import numpy as np
from sklearn.ensemble import IsolationForest


def train_improved(X, contamination=0.05, random_state=42):
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        max_samples='auto',
        bootstrap=True
    )
    model.fit(X)
    return model


def predict_improved(model, X):
    preds = model.predict(X)
    scores = model.decision_function(X)
    anomalies = np.where(preds == -1)[0]
    return anomalies, scores, preds
