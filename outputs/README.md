# outputs/

Resultados de la detección de anomalías.

## Subdirectorios

- `baseline/` — Resultados del modelo baseline (contamination automática)
- `improved/` — Resultados del modelo mejorado (hiperparámetros optimizados)
- `notebook/` — Figuras generadas por el caso práctico (notebook)

Cada subdirectorio baseline/improved contiene:
- `anomalies.csv` — Registro de anomalías detectadas
- `plot.png` — Visualización de los resultados

## Figuras del Caso Práctico (`notebook/`)

| Archivo | Descripción |
|---|---|
| `01_eda.png` | Histogramas y boxplots por variable (Normal vs Anomalía) |
| `02_scatter_ground_truth.png` | Scatter 2D con ground truth |
| `03_score_distribution.png` | Distribución de scores de anomalía |
| `04_score_vs_ground_truth.png` | Scores vs clase real + umbral |
| `05_confusion_matrices.png` | Matrices de confusión (Baseline vs Mejorado) |
| `06_predictions_scatter.png` | Mapa TP / FP / FN / TN en espacio 2D |
| `07_time_series.png` | Serie temporal: ground truth vs detectado |
| `08_contamination_sensitivity.png` | Efecto del parámetro contamination |
| `09_n_estimators_convergence.png` | Convergencia por número de árboles |
