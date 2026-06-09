import os
from src.preprocess import load_data, preprocess
from src.train_improved import train_improved, predict_improved
from src.evaluate import evaluate, save_anomalies
from src.visualize import plot_anomalies


DATA_PATH = os.path.join('data', 'synthetic_iot_dataset_challenging.csv')
OUTPUT_DIR = os.path.join('outputs', 'improved')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("[IMPROVED] Cargando datos...")
    df = load_data(DATA_PATH)
    print(f"[IMPROVED] Datos cargados: {df.shape}")

    X, y, _ = preprocess(df)
    print(f"[IMPROVED] Features: {X.shape[1]}")

    model = train_improved(X, contamination=0.05)
    anomalies, scores, preds = predict_improved(model, X)
    print(f"[IMPROVED] Anomalías detectadas: {len(anomalies)}")

    metrics = evaluate(anomalies, preds, scores, y)
    print(f"[IMPROVED] Tasa de anomalía: {metrics['anomaly_rate']:.4f}")

    save_anomalies(df, anomalies, os.path.join(OUTPUT_DIR, 'anomalies.csv'))
    print(f"[IMPROVED] Anomalías guardadas en {OUTPUT_DIR}/anomalies.csv")

    plot_anomalies(X, anomalies, os.path.join(OUTPUT_DIR, 'plot.png'))
    print(f"[IMPROVED] Gráfico guardado en {OUTPUT_DIR}/plot.png")


if __name__ == '__main__':
    main()
