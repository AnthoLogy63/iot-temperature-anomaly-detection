import numpy as np
import pandas as pd


def evaluate(anomalies, preds, scores, y_true):
    results = {
        'total_samples': len(preds),
        'anomalies_detected': len(anomalies),
        'anomaly_rate': len(anomalies) / len(preds)
    }

    if y_true is not None:
        tp = np.sum((preds == -1) & (y_true == 1))
        fp = np.sum((preds == -1) & (y_true == 0))
        fn = np.sum((preds == 1) & (y_true == 1))
        tn = np.sum((preds == 1) & (y_true == 0))

        results['true_positives'] = int(tp)
        results['false_positives'] = int(fp)
        results['false_negatives'] = int(fn)
        results['true_negatives'] = int(tn)
        results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['f1_score'] = (
            2 * results['precision'] * results['recall'] /
            (results['precision'] + results['recall'])
            if (results['precision'] + results['recall']) > 0 else 0
        )

    return results


def save_anomalies(df, anomalies, filepath):
    anomaly_df = df.iloc[anomalies].copy()
    anomaly_df['anomaly'] = 1
    anomaly_df.to_csv(filepath, index=False)
