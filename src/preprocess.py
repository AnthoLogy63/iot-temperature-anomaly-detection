import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def preprocess(df, feature_cols=None, scale=True):
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in feature_cols:
            feature_cols.remove('label')
        if 'Label' in feature_cols:
            feature_cols.remove('Label')

    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_cols)

    y = None
    for col in ['label', 'Label', 'anomaly', 'Anomaly']:
        if col in df.columns:
            y = df[col]
            break

    return X, y, feature_cols
