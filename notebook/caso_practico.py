"""
Caso Practico: Isolation Forest para Deteccion de Anomalias en Sensores IoT
Equivalente al contenido del notebook caso_practico.ipynb.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# Estilo global de graficas
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 11
PALETTE = {'normal': '#2196F3', 'anomaly': '#F44336'}

OUTPUT_DIR = os.path.join('outputs', 'notebook')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"NumPy {np.__version__} | Pandas {pd.__version__}")

# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join('data', 'synthetic_iot_dataset_challenging.csv')
df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("DATASET: Sensores IoT de Temperatura/Humedad")
print("=" * 55)
print(f"\nDimensiones : {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nColumnas:\n{df.dtypes.to_string()}")
print(f"\nPrimeras filas:")
print(df.head(4).to_string())
print(f"\nValores faltantes: {df.isnull().sum().sum()}")

vc = df['Anomaly'].value_counts()
print(f"\nDistribucion de etiquetas:")
print(f"  Normal   (0): {vc[0]:>4d} ({vc[0]/len(df)*100:.1f}%)")
print(f"  Anomalia (1): {vc[1]:>4d} ({vc[1]/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Analisis Exploratorio (EDA)
# ---------------------------------------------------------------------------
FEATURES = ['Temperature', 'Humidity', 'Battery_Level']
X = df[FEATURES].copy()
y = df['Anomaly'].copy()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Analisis Exploratorio del Dataset IoT', fontsize=14, fontweight='bold', y=1.01)

for i, col in enumerate(FEATURES):
    ax = axes[0, i]
    ax.hist(X[col][y == 0], bins=40, alpha=0.65, color=PALETTE['normal'], label='Normal', density=True)
    ax.hist(X[col][y == 1], bins=40, alpha=0.65, color=PALETTE['anomaly'], label='Anomalia', density=True)
    ax.set_title(f'Distribucion de {col}', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Densidad')
    ax.legend()

for i, col in enumerate(FEATURES):
    ax = axes[1, i]
    data_to_plot = [X[col][y == 0].values, X[col][y == 1].values]
    bp = ax.boxplot(data_to_plot, labels=['Normal', 'Anomalia'],
                    patch_artist=True, boxprops=dict(linewidth=1.5))
    bp['boxes'][0].set_facecolor(PALETTE['normal'] + '80')
    bp['boxes'][1].set_facecolor(PALETTE['anomaly'] + '80')
    ax.set_title(f'Boxplot de {col}', fontweight='bold')
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_eda.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figura 1: 01_eda.png")

pairs = [('Temperature', 'Humidity'), ('Temperature', 'Battery_Level'), ('Humidity', 'Battery_Level')]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Relaciones entre Variables (Ground Truth)', fontsize=14, fontweight='bold')
for ax, (x_col, y_col) in zip(axes, pairs):
    ax.scatter(X[x_col][y==0], X[y_col][y==0], c=PALETTE['normal'], alpha=0.3, s=15, label='Normal')
    ax.scatter(X[x_col][y==1], X[y_col][y==1], c=PALETTE['anomaly'], alpha=0.7, s=25, label='Anomalia', zorder=3)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}')
    ax.legend(markerscale=1.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_scatter_ground_truth.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figura 2: 02_scatter_ground_truth.png")

# ---------------------------------------------------------------------------
# Fundamentos matematicos
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("FUNDAMENTOS MATEMATICOS")
print("=" * 55)

def euler_mascheroni_H(i):
    """Numero armonico H(i) = sum(1/k) para k=1..i"""
    return sum(1.0/k for k in range(1, i+1))

def c_n(n):
    """
    Longitud media de busqueda fallida en un BST de n nodos.
    Funcion de normalizacion de Isolation Forest.
    c(n) = 2 * H(n-1) - 2*(n-1)/n
    """
    if n <= 1:
        return 1.0
    return 2 * euler_mascheroni_H(n - 1) - 2*(n-1)/n

def anomaly_score(h_x_expected, n):
    """
    Puntuacion de anomalia: s(x, n) = 2^(-E[h(x)] / c(n))
    s -> 1.0 : anomalia  |  s -> 0.5 : ambiguo  |  s -> 0.0 : normal
    """
    return 2 ** (-h_x_expected / c_n(n))

n_demo = 256
c = c_n(n_demo)
print(f"\nPara n = {n_demo} (max_samples por defecto):")
print(f"  H(n-1) = H(255) = {euler_mascheroni_H(255):.4f}")
print(f"  c(n)   = {c:.4f}")
print(f"\n  {'h(x)':>8} | {'s(x,n)':>10} | {'Clasificacion':>15}")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*15}")
for h_val in [1, 3, 5, 8, 10, c, 12, 14, 16]:
    s = anomaly_score(h_val, n_demo)
    label = "ANOMALIA" if s > 0.7 else ("Ambiguo" if s > 0.45 else "Normal")
    print(f"  {h_val:>8.1f} | {s:>10.4f} | {label:>15}")
print(f"\n  c(n) = {c:.4f} -> cuando h(x) = c(n), s = 2^(-1) = 0.5 (punto medio)")

# ---------------------------------------------------------------------------
# Modelo Baseline
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("MODELO BASELINE")
print("=" * 55)
print("""
Parametros:
  n_estimators  = 100
  contamination = 'auto'  (umbral teorico: score < 0 -> anomalia)
  max_samples   = 'auto'  (min(256, n_samples))
  random_state  = 42
""")

model_baseline = IsolationForest(
    n_estimators=100,
    contamination='auto',
    max_samples='auto',
    random_state=42
)
model_baseline.fit(X)

preds_b    = model_baseline.predict(X)
scores_b   = model_baseline.score_samples(X)
decision_b = model_baseline.decision_function(X)

anomalies_b = np.where(preds_b == -1)[0]
print(f"Anomalias detectadas: {len(anomalies_b)} ({len(anomalies_b)/len(X)*100:.1f}%)")
print(f"Rango de scores: [{decision_b.min():.4f}, {decision_b.max():.4f}]")
print("Nota: decision_function() < 0 indica anomalia, > 0 indica normal.")

# ---------------------------------------------------------------------------
# Modelo Mejorado
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("MODELO MEJORADO")
print("=" * 55)
print("""
Parametros:
  n_estimators  = 200  (mayor estabilidad)
  contamination = 0.05 (umbral en el percentil 5 de los scores)
  max_samples   = 256  (valor del paper original)
  bootstrap     = False
  random_state  = 42
""")

model_improved = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    max_samples=256,
    bootstrap=False,
    random_state=42
)
model_improved.fit(X)

preds_i    = model_improved.predict(X)
scores_i   = model_improved.score_samples(X)
decision_i = model_improved.decision_function(X)

anomalies_i = np.where(preds_i == -1)[0]
print(f"Anomalias detectadas: {len(anomalies_i)} ({len(anomalies_i)/len(X)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Modelo Optimo (contamination ajustado a la tasa real)
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("MODELO OPTIMO (contamination=0.174)")
print("=" * 55)
print("""
Tasa real de anomalias en el dataset: 17.4%
contamination ajustado a ese valor para comparar efecto del parametro.
""")

model_optimal = IsolationForest(
    n_estimators=200,
    contamination=0.174,
    max_samples=256,
    bootstrap=False,
    random_state=42
)
model_optimal.fit(X)

preds_o     = model_optimal.predict(X)
decision_o  = model_optimal.decision_function(X)
anomalies_o = np.where(preds_o == -1)[0]
print(f"Anomalias detectadas: {len(anomalies_o)} ({len(anomalies_o)/len(X)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Distribucion de scores
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Distribucion de Puntuaciones de Anomalia', fontsize=14, fontweight='bold')

for ax, (scores, preds, title) in zip(axes, [
    (decision_b, preds_b, 'Baseline (contamination="auto")'),
    (decision_i, preds_i, 'Mejorado (contamination=0.05)')
]):
    ax.hist(scores[preds == 1],  bins=50, alpha=0.7, color=PALETTE['normal'],
            label='Normal predicho', density=True)
    ax.hist(scores[preds == -1], bins=50, alpha=0.7, color=PALETTE['anomaly'],
            label='Anomalia predicha', density=True)
    ax.axvline(0.0, color='black', linewidth=2, linestyle='--', label='Umbral = 0')
    ax.set_xlabel('decision_function(x)')
    ax.set_ylabel('Densidad')
    ax.set_title(title, fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_score_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figura 3: 03_score_distribution.png")

# Score continuo vs ground truth
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(decision_i[y == 0], bins=50, alpha=0.65, color=PALETTE['normal'],
        label='Normal (ground truth)', density=True)
ax.hist(decision_i[y == 1], bins=50, alpha=0.65, color=PALETTE['anomaly'],
        label='Anomalia (ground truth)', density=True)
ax.axvline(0.0, color='black', linewidth=2, linestyle='--', label='Umbral (score = 0)')
q5 = np.percentile(decision_i, 5)
ax.axvline(q5, color='orange', linewidth=2, linestyle=':', label=f'Percentil 5% = {q5:.3f}')
ax.set_xlabel('Puntuacion de Anomalia - decision_function(x)')
ax.set_ylabel('Densidad')
ax.set_title('Distribucion de Scores: Normal vs Anomalia (Ground Truth)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_score_vs_ground_truth.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figura 4: 04_score_vs_ground_truth.png")

# ---------------------------------------------------------------------------
# Evaluacion de metricas
# ---------------------------------------------------------------------------
def evaluate_model(name, preds, scores, y_true):
    """Calcula precision, recall, F1 y ROC-AUC. Devuelve dict con resultados."""
    # IF usa -1/+1; ground truth usa 1/0
    preds_binary = (preds == -1).astype(int)

    tp = int(np.sum((preds_binary == 1) & (y_true == 1)))
    fp = int(np.sum((preds_binary == 1) & (y_true == 0)))
    fn = int(np.sum((preds_binary == 0) & (y_true == 1)))
    tn = int(np.sum((preds_binary == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # Negamos scores porque decision_function da valores mas negativos a anomalias
    roc_auc   = roc_auc_score(y_true, -scores)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  TP={tp:>4}  FP={fp:>4}  FN={fn:>4}  TN={tn:>4}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  Anomalias detectadas: {tp+fp} ({(tp+fp)/len(preds)*100:.1f}%)")
    print(f"  Ground truth anomalias: {int(y_true.sum())} ({y_true.mean()*100:.1f}%)")

    return {
        'name': name, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'roc_auc': roc_auc, 'preds_binary': preds_binary
    }

print("\n" + "=" * 55)
print("EVALUACION DE METRICAS")
print("=" * 55)
metrics_b = evaluate_model("Baseline (contamination='auto')", preds_b, decision_b, y)
metrics_i = evaluate_model("Mejorado (contamination=0.05)",   preds_i, decision_i, y)
metrics_o = evaluate_model("Optimo   (contamination=0.174)",  preds_o, decision_o, y)

# ---------------------------------------------------------------------------
# Matrices de confusion (3 modelos)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
fig.suptitle('Matrices de Confusion: Comparacion de los 3 Modelos', fontsize=14, fontweight='bold')

for ax, metrics, title in zip(axes,
    [metrics_b, metrics_i, metrics_o],
    ["Baseline\n(contamination='auto')",
     'Mejorado\n(contamination=0.05)',
     'Optimo\n(contamination=0.174)']):
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomalia'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{title}\nF1={metrics["f1"]:.3f} | AUC={metrics["roc_auc"]:.3f}', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figura 5: 05_confusion_matrices.png (3 modelos)")

# ---------------------------------------------------------------------------
# Scatter: Ground Truth vs Modelo Optimo (TP/FP/FN/TN)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Predicciones del Modelo Optimo (contamination=0.174): Temperature vs Humidity',
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.scatter(X['Temperature'][y==0], X['Humidity'][y==0],
           c=PALETTE['normal'], alpha=0.3, s=15, label='Normal (GT)')
ax.scatter(X['Temperature'][y==1], X['Humidity'][y==1],
           c=PALETTE['anomaly'], alpha=0.7, s=25, label='Anomalia (GT)', zorder=3)
ax.set_title('Ground Truth (522 anomalias reales)', fontweight='bold')
ax.set_xlabel('Temperature (norm.)')
ax.set_ylabel('Humidity (norm.)')
ax.legend()

ax = axes[1]
preds_bin_o = metrics_o['preds_binary']
y_arr       = y.values

tp_mask = (preds_bin_o == 1) & (y_arr == 1)
fp_mask = (preds_bin_o == 1) & (y_arr == 0)
fn_mask = (preds_bin_o == 0) & (y_arr == 1)
tn_mask = (preds_bin_o == 0) & (y_arr == 0)

ax.scatter(X['Temperature'][tn_mask], X['Humidity'][tn_mask],
           c='#2196F3', alpha=0.2, s=12, label=f'TN={tn_mask.sum()}')
ax.scatter(X['Temperature'][tp_mask], X['Humidity'][tp_mask],
           c='#4CAF50', alpha=0.8, s=30, label=f'TP={tp_mask.sum()}', marker='*', zorder=4)
ax.scatter(X['Temperature'][fp_mask], X['Humidity'][fp_mask],
           c='#FF9800', alpha=0.8, s=30, label=f'FP={fp_mask.sum()}', marker='^', zorder=4)
ax.scatter(X['Temperature'][fn_mask], X['Humidity'][fn_mask],
           c='#9C27B0', alpha=0.8, s=30, label=f'FN={fn_mask.sum()}', marker='x', zorder=4)
ax.set_title(f'Optimo: TP={tp_mask.sum()}  FP={fp_mask.sum()}  FN={fn_mask.sum()}', fontweight='bold')
ax.set_xlabel('Temperature (norm.)')
ax.set_ylabel('Humidity (norm.)')
ax.legend(markerscale=1.5, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_predictions_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figura 6: 06_predictions_scatter.png (modelo optimo)")

# ---------------------------------------------------------------------------
# Serie temporal simulada
# ---------------------------------------------------------------------------
temp_series       = X['Temperature'].values
anomaly_mask_gt   = y.values.astype(bool)
anomaly_mask_pred = (preds_i == -1)

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle('Serie Temporal de Temperatura - Sensor IoT', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(temp_series, color='#546E7A', alpha=0.6, linewidth=0.8, label='Temperatura')
ax.scatter(np.where(anomaly_mask_gt)[0], temp_series[anomaly_mask_gt],
           c=PALETTE['anomaly'], s=20, zorder=3, label='Anomalia real', alpha=0.9)
ax.set_title('Ground Truth (etiquetas reales)', fontweight='bold')
ax.set_ylabel('Temperatura (norm.)')
ax.legend()

ax = axes[1]
ax.plot(temp_series, color='#546E7A', alpha=0.6, linewidth=0.8, label='Temperatura')
ax.scatter(np.where(anomaly_mask_pred)[0], temp_series[anomaly_mask_pred],
           c='#FF9800', s=20, zorder=3, label='Anomalia detectada (IF)', alpha=0.9)
ax.set_title('Predicciones del Isolation Forest (contamination=0.05)', fontweight='bold')
ax.set_ylabel('Temperatura (norm.)')
ax.set_xlabel('Indice de tiempo')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_time_series.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figura 7: 07_time_series.png")

# ---------------------------------------------------------------------------
# Analisis de sensibilidad: efecto de contamination
# ---------------------------------------------------------------------------
print("\nCalculando sensibilidad al parametro contamination...")
contam_values    = np.arange(0.01, 0.35, 0.01)
f1_scores        = []
precision_scores = []
recall_scores    = []

for c_val in contam_values:
    m = IsolationForest(n_estimators=100, contamination=c_val, max_samples=256, random_state=42)
    m.fit(X)
    p_bin = (m.predict(X) == -1).astype(int)
    f1_scores.append(f1_score(y, p_bin, zero_division=0))
    precision_scores.append(precision_score(y, p_bin, zero_division=0))
    recall_scores.append(recall_score(y, p_bin, zero_division=0))

best_idx    = np.argmax(f1_scores)
best_contam = contam_values[best_idx]
best_f1     = f1_scores[best_idx]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(contam_values, f1_scores,        'b-o', markersize=4, linewidth=2, label='F1-Score')
ax.plot(contam_values, precision_scores, 'g--', linewidth=1.5, label='Precision')
ax.plot(contam_values, recall_scores,    'r--', linewidth=1.5, label='Recall')
ax.axvline(best_contam, color='orange', linewidth=2, linestyle=':',
           label=f'Mejor contamination = {best_contam:.2f}  (F1={best_f1:.3f})')
ax.axvline(0.05, color='purple', linewidth=2, linestyle='--',
           label='contamination=0.05 (modelo mejorado)')
ax.set_xlabel('Parametro contamination')
ax.set_ylabel('Metrica')
ax.set_title('Efecto del Parametro contamination sobre F1, Precision y Recall', fontweight='bold')
ax.legend()
ax.set_xlim(0, 0.35)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_contamination_sensitivity.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Figura 8: 08_contamination_sensitivity.png")
print(f"  Mejor contamination: {best_contam:.2f}  ->  F1 = {best_f1:.4f}")

# ---------------------------------------------------------------------------
# Convergencia: efecto de n_estimators
# ---------------------------------------------------------------------------
print("\nCalculando efecto de n_estimators...")
n_trees  = [10, 25, 50, 100, 150, 200, 300, 400, 500]
f1_trees = []
for n in n_trees:
    m = IsolationForest(n_estimators=n, contamination=0.05, max_samples=256, random_state=42)
    m.fit(X)
    p = (m.predict(X) == -1).astype(int)
    f1_trees.append(f1_score(y, p, zero_division=0))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(n_trees, f1_trees, 'b-o', markersize=6, linewidth=2)
ax.axvline(200, color='orange', linewidth=2, linestyle='--', label='n_estimators=200 (modelo mejorado)')
ax.set_xlabel('Numero de arboles (n_estimators)')
ax.set_ylabel('F1-Score')
ax.set_title('Convergencia del Modelo: F1-Score vs Numero de Arboles', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_n_estimators_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figura 9: 09_n_estimators_convergence.png")

# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("RESUMEN FINAL DEL EXPERIMENTO - 3 MODELOS")
print("=" * 75)

models = [metrics_b, metrics_i, metrics_o]
print(f"Dataset: {DATA_PATH}")
print(f"  Registros: {len(df):,}  |  Anomalias reales: {int(y.sum())} ({y.mean()*100:.1f}%)")
print(f"\n{'Metrica':<28} {'Baseline':>12} {'Mejorado':>12} {'Optimo':>12}")
print("-" * 68)
for key, label in [('precision','Precision'), ('recall','Recall'),
                   ('f1','F1-Score'), ('roc_auc','ROC-AUC')]:
    print(f"  {label:<26} {models[0][key]:>12.4f} {models[1][key]:>12.4f} {models[2][key]:>12.4f}")
print("-" * 68)
for key, label in [('tp','TP'), ('fp','FP (falsas alarmas)'), ('fn','FN (no detectadas)')]:
    print(f"  {label:<26} {models[0][key]:>12} {models[1][key]:>12} {models[2][key]:>12}")
print("-" * 68)
print(f"  {'Anomalias detectadas':<26} "
      f"{models[0]['tp']+models[0]['fp']:>12} "
      f"{models[1]['tp']+models[1]['fp']:>12} "
      f"{models[2]['tp']+models[2]['fp']:>12}")
print("\nConclusiones:")
print("  AUC ~ 0.55: las anomalias se superponen con los datos normales")
print("  en el espacio de features (Temperature, Humidity, Battery_Level).")
print("  Dataset disenado como 'challenging' (dificil de separar).")
print("  Isolation Forest funciona mejor cuando las anomalias estan")
print("  claramente separadas en alguna dimension del espacio de features.")
print()
print("Script completado. Figuras guardadas en:", OUTPUT_DIR)
