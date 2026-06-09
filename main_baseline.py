import os
from src.preprocess import load_data, preprocess
from src.train_baseline import train_baseline, predict_baseline
from src.evaluate import evaluate, save_anomalies
from src.visualize import plot_anomalies


DATA_PATH = os.path.join('data', 'clean_unlabeled_autoencoder.csv')
OUTPUT_DIR = os.path.join('outputs', 'baseline')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("[BASELINE] Cargando datos...")
    df = load_data(DATA_PATH)
    print(f"[BASELINE] Datos cargados: {df.shape}")

    X, y, _ = preprocess(df)
    print(f"[BASELINE] Features: {X.shape[1]}")

    model = train_baseline(X)
    anomalies, scores, preds = predict_baseline(model, X)
    print(f"[BASELINE] Anomalías detectadas: {len(anomalies)}")

    metrics = evaluate(anomalies, preds, scores, y)
    print(f"[BASELINE] Tasa de anomalía: {metrics['anomaly_rate']:.4f}")

    save_anomalies(df, anomalies, os.path.join(OUTPUT_DIR, 'anomalies.csv'))
    print(f"[BASELINE] Anomalías guardadas en {OUTPUT_DIR}/anomalies.csv")

    plot_anomalies(X, anomalies, os.path.join(OUTPUT_DIR, 'plot.png'))
    print(f"[BASELINE] Gráfico guardado en {OUTPUT_DIR}/plot.png")


if __name__ == '__main__':
    main()
