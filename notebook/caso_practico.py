"""
Caso Práctico: Isolation Forest para Detección de Anomalías en Sensores IoT
============================================================================
Este script es equivalente al contenido del notebook caso_practico.ipynb.
Se usa para verificar que todo el código funciona correctamente.
"""

# =============================================================================
# CELDA 1: Importaciones y Configuración
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# Estilo visual global
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 11
PALETTE = {'normal': '#2196F3', 'anomaly': '#F44336'}

import os
OUTPUT_DIR = os.path.join('outputs', 'notebook')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ Librerías cargadas correctamente.")
print(f"   NumPy {np.__version__} | Pandas {pd.__version__}")

# =============================================================================
# CELDA 2: Carga y Exploración del Dataset
# =============================================================================
DATA_PATH = os.path.join('data', 'synthetic_iot_dataset_challenging.csv')
df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("DATASET: Sensores IoT de Temperatura/Humedad")
print("=" * 55)
print(f"\n📦 Dimensiones : {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"\n📋 Columnas:\n{df.dtypes.to_string()}")
print(f"\n🔍 Primeras filas:")
print(df.head(4).to_string())
print(f"\n✅ Valores faltantes: {df.isnull().sum().sum()} (ninguno)")
print(f"\n🏷️  Distribución de etiquetas:")
vc = df['Anomaly'].value_counts()
print(f"   Normal    (0): {vc[0]:>4d} ({vc[0]/len(df)*100:.1f}%)")
print(f"   Anomalía  (1): {vc[1]:>4d} ({vc[1]/len(df)*100:.1f}%)")

# =============================================================================
# CELDA 3: Análisis Exploratorio de Datos (EDA)
# =============================================================================
FEATURES = ['Temperature', 'Humidity', 'Battery_Level']
X = df[FEATURES].copy()
y = df['Anomaly'].copy()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Análisis Exploratorio del Dataset IoT', fontsize=14, fontweight='bold', y=1.01)

# Fila 1: Histogramas
for i, col in enumerate(FEATURES):
    ax = axes[0, i]
    ax.hist(X[col][y == 0], bins=40, alpha=0.65, color=PALETTE['normal'], label='Normal', density=True)
    ax.hist(X[col][y == 1], bins=40, alpha=0.65, color=PALETTE['anomaly'], label='Anomalía', density=True)
    ax.set_title(f'Distribución de {col}', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Densidad')
    ax.legend()

# Fila 2: Boxplots
for i, col in enumerate(FEATURES):
    ax = axes[1, i]
    data_to_plot = [X[col][y == 0].values, X[col][y == 1].values]
    bp = ax.boxplot(data_to_plot, labels=['Normal', 'Anomalía'],
                    patch_artist=True,
                    boxprops=dict(linewidth=1.5))
    bp['boxes'][0].set_facecolor(PALETTE['normal'] + '80')
    bp['boxes'][1].set_facecolor(PALETTE['anomaly'] + '80')
    ax.set_title(f'Boxplot de {col}', fontweight='bold')
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_eda.png'), dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figura 1 guardada: 01_eda.png")

# Scatter plots de features
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Relaciones entre Variables (Ground Truth)', fontsize=14, fontweight='bold')
pairs = [('Temperature', 'Humidity'), ('Temperature', 'Battery_Level'), ('Humidity', 'Battery_Level')]
for ax, (x_col, y_col) in zip(axes, pairs):
    ax.scatter(X[x_col][y==0], X[y_col][y==0], c=PALETTE['normal'], alpha=0.3, s=15, label='Normal')
    ax.scatter(X[x_col][y==1], X[y_col][y==1], c=PALETTE['anomaly'], alpha=0.7, s=25, label='Anomalía', zorder=3)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}')
    ax.legend(markerscale=1.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_scatter_ground_truth.png'), dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figura 2 guardada: 02_scatter_ground_truth.png")

# =============================================================================
# CELDA 4: La Matemática del Isolation Forest
# =============================================================================
print("\n" + "=" * 55)
print("FUNDAMENTOS MATEMÁTICOS")
print("=" * 55)

def euler_mascheroni_H(i):
    """Número armónico H(i) ≈ ln(i) + gamma"""
    # Fórmula exacta: suma 1/k para k de 1 a i
    return sum(1.0/k for k in range(1, i+1))

def c_n(n):
    """
    Longitud media de búsqueda fallida en un BST (Binary Search Tree).
    Esta es la función de normalización de Isolation Forest.
    
    c(n) = 2 * H(n-1) - 2*(n-1)/n
    
    Donde H(i) es el i-ésimo número armónico:
    H(i) = ln(i) + 0.5772... (constante de Euler-Mascheroni γ)
    """
    if n <= 1:
        return 1.0
    H_n_minus_1 = euler_mascheroni_H(n - 1)
    return 2 * H_n_minus_1 - 2*(n-1)/n

def anomaly_score(h_x_expected, n):
    """
    Puntuación de anomalía normalizada.
    
    s(x, n) = 2^{ -E[h(x)] / c(n) }
    
    - s ≈ 1.0 → anomalía (h(x) muy corto, aislado fácilmente)
    - s ≈ 0.5 → dato ambiguo
    - s ≈ 0.0 → dato normal (h(x) muy largo, difícil de aislar)
    """
    return 2 ** (-h_x_expected / c_n(n))

# Demostración con n = 256 (default max_samples de sklearn)
n = 256
c = c_n(n)
print(f"\n📐 Para n = {n} (max_samples default):")
print(f"   H(n-1) = H(255) ≈ {euler_mascheroni_H(255):.4f}")
print(f"   c(n)   = 2·H(n-1) - 2(n-1)/n = {c:.4f}")
print(f"\n📊 Tabla de puntuaciones según longitud del camino h(x):")
print(f"   {'h(x)':>8} | {'s(x, n)':>10} | {'Interpretación':>20}")
print(f"   {'-'*8}-+-{'-'*10}-+-{'-'*20}")
for h_val in [1, 3, 5, 8, 10, c, 12, 14, 16]:
    s = anomaly_score(h_val, n)
    if s > 0.7:
        label = "🔴 ANOMALÍA"
    elif s > 0.45:
        label = "🟡 Ambiguo"
    else:
        label = "🔵 Normal"
    print(f"   {h_val:>8.1f} | {s:>10.4f} | {label:>20}")

print(f"\n   💡 c(n) = {c:.4f} → cuando h(x) = c(n), s = 2^(-1) = 0.5 (punto medio)")

# =============================================================================
# CELDA 5: Entrenamiento — Modelo Baseline
# =============================================================================
print("\n" + "=" * 55)
print("MODELO BASELINE")
print("=" * 55)
print("""
Parámetros:
  n_estimators  = 100     → 100 árboles en el bosque
  contamination = 'auto'  → umbral basado en la teoría (score < 0 → anomalía)
  max_samples   = 'auto'  → min(256, n_samples)
  random_state  = 42      → reproducibilidad
""")

model_baseline = IsolationForest(
    n_estimators=100,
    contamination='auto',   # No asume ninguna tasa de anomalías
    max_samples='auto',     # sklearn: min(256, n_samples)
    random_state=42
)
model_baseline.fit(X)

# Predicciones
preds_b = model_baseline.predict(X)           # -1 = anomalía, +1 = normal
scores_b = model_baseline.score_samples(X)    # = decision_function(X) + offset
decision_b = model_baseline.decision_function(X)  # Score crudo (negativo → anomalía)

anomalies_b = np.where(preds_b == -1)[0]
print(f"✅ Modelo baseline entrenado.")
print(f"   Anomalías detectadas: {len(anomalies_b)} ({len(anomalies_b)/len(X)*100:.1f}%)")
print(f"   Rango de scores:      [{decision_b.min():.4f}, {decision_b.max():.4f}]")
print(f"\n💡 Nota: decision_function() devuelve el score desplazado.")
print(f"   Valores < 0 → anomalía. Valores > 0 → normal.")

# =============================================================================
# CELDA 6: Entrenamiento — Modelo Mejorado
# =============================================================================
print("\n" + "=" * 55)
print("MODELO MEJORADO")
print("=" * 55)
print("""
Parámetros ajustados:
  n_estimators  = 200        → más árboles = más estabilidad
  contamination = 0.05       → asumimos que el 5% de los datos son anomalías
                               (umbral en el percentil 5 de los scores)
  max_samples   = 256        → submuestra fija (valor del paper original)
  random_state  = 42         → reproducibilidad
  bootstrap     = False      → sin reemplazo (más representatividad por árbol)
""")

model_improved = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # aún 5% para comparación con baseline
    max_samples=256,
    bootstrap=False,
    random_state=42
)
model_improved.fit(X)

preds_i = model_improved.predict(X)
scores_i = model_improved.score_samples(X)
decision_i = model_improved.decision_function(X)

anomalies_i = np.where(preds_i == -1)[0]
print(f"✅ Modelo mejorado entrenado.")
print(f"   Anomalías detectadas: {len(anomalies_i)} ({len(anomalies_i)/len(X)*100:.1f}%)")

# ── Modelo Óptimo (contamination ajustado a tasa real) ────────────────────
print("\n" + "=" * 55)
print("MODELO ÓPTIMO (contamination ajustado)")
print("=" * 55)
print("""
La tasa real de anomalías en el dataset es 17.4%.
Si conocemos esto a priori, podemos ajustar contamination para maximizar F1.
""")

model_optimal = IsolationForest(
    n_estimators=200,
    contamination=0.174,   # ajustado a la tasa real del dataset
    max_samples=256,
    bootstrap=False,
    random_state=42
)
model_optimal.fit(X)
preds_o    = model_optimal.predict(X)
decision_o = model_optimal.decision_function(X)
anomalies_o = np.where(preds_o == -1)[0]
print(f"✅ Modelo óptimo entrenado.")
print(f"   Anomalías detectadas: {len(anomalies_o)} ({len(anomalies_o)/len(X)*100:.1f}%)")
print(f"   Rango de scores:      [{decision_i.min():.4f}, {decision_i.max():.4f}]")

# =============================================================================
# CELDA 7: Visualización de Scores y Umbral
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Distribución de Puntuaciones de Anomalía', fontsize=14, fontweight='bold')

for ax, (scores, preds, title) in zip(axes, [
    (decision_b, preds_b, 'Baseline (contamination="auto")'),
    (decision_i, preds_i, 'Mejorado (contamination=0.05)')
]):
    # Histograma coloreado por predicción
    scores_normal = scores[preds == 1]
    scores_anom = scores[preds == -1]
    ax.hist(scores_normal, bins=50, alpha=0.7, color=PALETTE['normal'], label='Normal predicho', density=True)
    ax.hist(scores_anom, bins=50, alpha=0.7, color=PALETTE['anomaly'], label='Anomalía predicha', density=True)
    
    # Umbral (score = 0 en decision_function)
    threshold = 0.0
    ax.axvline(threshold, color='black', linewidth=2, linestyle='--', label=f'Umbral = {threshold}')
    ax.set_xlabel('decision_function(x)')
    ax.set_ylabel('Densidad')
    ax.set_title(title, fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_score_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figura 3 guardada: 03_score_distribution.png")

# Histograma del score CONTINUO vs ground truth (mejor visualización)
fig, ax = plt.subplots(figsize=(10, 5))
scores_normal_gt = decision_i[y == 0]
scores_anom_gt = decision_i[y == 1]
ax.hist(scores_normal_gt, bins=50, alpha=0.65, color=PALETTE['normal'], label='Normal (ground truth)', density=True)
ax.hist(scores_anom_gt, bins=50, alpha=0.65, color=PALETTE['anomaly'], label='Anomalía (ground truth)', density=True)
ax.axvline(0.0, color='black', linewidth=2, linestyle='--', label='Umbral (score = 0)')

# Cuantil 5%
q5 = np.percentile(decision_i, 5)
ax.axvline(q5, color='orange', linewidth=2, linestyle=':', label=f'Percentil 5% = {q5:.3f}')
ax.set_xlabel('Puntuación de Anomalía — decision_function(x)')
ax.set_ylabel('Densidad')
ax.set_title('Distribución de Scores: Normal vs Anomalía (Ground Truth)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_score_vs_ground_truth.png'), dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figura 4 guardada: 04_score_vs_ground_truth.png")

# =============================================================================
# CELDA 8: Evaluación de Métricas
# =============================================================================
def evaluate_model(name, preds, scores, y_true):
    """Calcula y muestra métricas de evaluación."""
    # Convertir: IF usa -1/+1; etiquetas ground truth son 1/0
    # Anomalía: IF=-1 ↔ ground truth=1
    preds_binary = (preds == -1).astype(int)
    
    tp = int(np.sum((preds_binary == 1) & (y_true == 1)))
    fp = int(np.sum((preds_binary == 1) & (y_true == 0)))
    fn = int(np.sum((preds_binary == 0) & (y_true == 1)))
    tn = int(np.sum((preds_binary == 0) & (y_true == 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # ROC-AUC usando score continuo (negado porque scores más negativos = más anómalo)
    roc_auc = roc_auc_score(y_true, -scores)
    
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  TP={tp:>4}  FP={fp:>4}  FN={fn:>4}  TN={tn:>4}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  Anomalías detectadas: {tp+fp} ({(tp+fp)/len(preds)*100:.1f}%)")
    print(f"  (Ground truth anomalías: {int(y_true.sum())} = {y_true.mean()*100:.1f}%)")
    
    return {'name': name, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
            'preds_binary': preds_binary}

print("\n" + "=" * 55)
print("EVALUACIÓN DE MÉTRICAS")
print("=" * 55)
metrics_b = evaluate_model("Baseline (contamination='auto')", preds_b, decision_b, y)
metrics_i = evaluate_model("Mejorado (contamination=0.05)",  preds_i, decision_i, y)
metrics_o = evaluate_model("Óptimo (contamination=0.174)",   preds_o, decision_o, y)

# =============================================================================
# CELDA 9: Matrices de Confusión (los 3 modelos)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
fig.suptitle('Matrices de Confusión: Comparación de los 3 Modelos', fontsize=14, fontweight='bold')

for ax, metrics, title in zip(axes,
    [metrics_b, metrics_i, metrics_o],
    ["Baseline\n(contamination='auto')",
     'Mejorado\n(contamination=0.05)',
     'Óptimo\n(contamination=0.174)']):
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomalía'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{title}\nF1={metrics["f1"]:.3f} | AUC={metrics["roc_auc"]:.3f}', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figura 5 guardada: 05_confusion_matrices.png (3 modelos)")

# =============================================================================
# CELDA 10: Scatter Plot — Comparación Ground Truth vs Óptimo
# =============================================================================
# Usamos el modelo Óptimo (contamination=0.174) porque tiene la misma
# cantidad de predicciones que anomalías reales → comparación más justa
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Predicciones del Modelo Óptimo (contamination=0.174): Temperature vs Humidity',
             fontsize=13, fontweight='bold')

# Panel izquierdo: Ground Truth
ax = axes[0]
ax.scatter(X['Temperature'][y==0], X['Humidity'][y==0],
           c=PALETTE['normal'], alpha=0.3, s=15, label='Normal (GT)')
ax.scatter(X['Temperature'][y==1], X['Humidity'][y==1],
           c=PALETTE['anomaly'], alpha=0.7, s=25, label='Anomalía (GT)', zorder=3)
ax.set_title('Ground Truth (522 anomalías reales)', fontweight='bold')
ax.set_xlabel('Temperature (norm.)')
ax.set_ylabel('Humidity (norm.)')
ax.legend()

# Panel derecho: TP, FP, FN, TN — Modelo Óptimo
ax = axes[1]
preds_bin_o = metrics_o['preds_binary']
y_arr = y.values

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
ax.set_title(f'Óptimo: TP={tp_mask.sum()} FP={fp_mask.sum()} FN={fn_mask.sum()}', fontweight='bold')
ax.set_xlabel('Temperature (norm.)')
ax.set_ylabel('Humidity (norm.)')
ax.legend(markerscale=1.5, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_predictions_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figura 6 guardada: 06_predictions_scatter.png (modelo óptimo)")

# =============================================================================
# CELDA 11: Serie Temporal Simulada
# =============================================================================
# Tomamos la temperatura y la ordenamos como si fuera una serie temporal
temp_series = X['Temperature'].values
anomaly_mask_gt = y.values.astype(bool)
anomaly_mask_pred = (preds_i == -1)

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle('Serie Temporal de Temperatura — Sensor IoT', fontsize=14, fontweight='bold')

# Panel superior: Ground Truth
ax = axes[0]
ax.plot(temp_series, color='#546E7A', alpha=0.6, linewidth=0.8, label='Temperatura')
ax.scatter(np.where(anomaly_mask_gt)[0], temp_series[anomaly_mask_gt],
           c=PALETTE['anomaly'], s=20, zorder=3, label='Anomalía real', alpha=0.9)
ax.set_title('Ground Truth (etiquetas reales)', fontweight='bold')
ax.set_ylabel('Temperatura (norm.)')
ax.legend()

# Panel inferior: Predicciones del modelo
ax = axes[1]
ax.plot(temp_series, color='#546E7A', alpha=0.6, linewidth=0.8, label='Temperatura')
ax.scatter(np.where(anomaly_mask_pred)[0], temp_series[anomaly_mask_pred],
           c='#FF9800', s=20, zorder=3, label='Anomalía detectada (IF)', alpha=0.9)
ax.set_title('Predicciones del Isolation Forest (contamination=0.05)', fontweight='bold')
ax.set_ylabel('Temperatura (norm.)')
ax.set_xlabel('Índice de tiempo')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_time_series.png'), dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figura 7 guardada: 07_time_series.png")

# =============================================================================
# CELDA 12: Análisis de Sensibilidad — Efecto de contamination
# =============================================================================
print("\n⏳ Calculando sensibilidad al parámetro contamination...")
contam_values = np.arange(0.01, 0.35, 0.01)
f1_scores = []
precision_scores = []
recall_scores = []

for c_val in contam_values:
    m = IsolationForest(n_estimators=100, contamination=c_val, max_samples=256, random_state=42)
    m.fit(X)
    p = m.predict(X)
    p_bin = (p == -1).astype(int)
    f1 = f1_score(y, p_bin, zero_division=0)
    prec = precision_score(y, p_bin, zero_division=0)
    rec = recall_score(y, p_bin, zero_division=0)
    f1_scores.append(f1)
    precision_scores.append(prec)
    recall_scores.append(rec)

best_idx = np.argmax(f1_scores)
best_contam = contam_values[best_idx]
best_f1 = f1_scores[best_idx]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(contam_values, f1_scores, 'b-o', markersize=4, linewidth=2, label='F1-Score')
ax.plot(contam_values, precision_scores, 'g--', linewidth=1.5, label='Precision')
ax.plot(contam_values, recall_scores, 'r--', linewidth=1.5, label='Recall')
ax.axvline(best_contam, color='orange', linewidth=2, linestyle=':', 
           label=f'Mejor contamination = {best_contam:.2f} (F1={best_f1:.3f})')
ax.axvline(0.05, color='purple', linewidth=2, linestyle='--', label='contamination=0.05 (nuestro modelo)')
ax.set_xlabel('Parámetro contamination')
ax.set_ylabel('Métrica')
ax.set_title('Efecto del Parámetro contamination sobre F1, Precision y Recall', fontweight='bold')
ax.legend()
ax.set_xlim(0, 0.35)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_contamination_sensitivity.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"📊 Figura 8 guardada: 08_contamination_sensitivity.png")
print(f"   🏆 Mejor contamination: {best_contam:.2f} → F1 = {best_f1:.4f}")

# =============================================================================
# CELDA 13: Efecto de n_estimators (Convergencia)
# =============================================================================
print("\n⏳ Calculando efecto de n_estimators...")
n_trees = [10, 25, 50, 100, 150, 200, 300, 400, 500]
f1_trees = []
for n in n_trees:
    m = IsolationForest(n_estimators=n, contamination=0.05, max_samples=256, random_state=42)
    m.fit(X)
    p = (m.predict(X) == -1).astype(int)
    f1_trees.append(f1_score(y, p, zero_division=0))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(n_trees, f1_trees, 'b-o', markersize=6, linewidth=2)
ax.axvline(200, color='orange', linewidth=2, linestyle='--', label='n_estimators=200 (nuestro modelo)')
ax.set_xlabel('Número de árboles (n_estimators)')
ax.set_ylabel('F1-Score')
ax.set_title('Convergencia del Modelo: F1-Score vs Número de Árboles', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_n_estimators_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figura 9 guardada: 09_n_estimators_convergence.png")

# =============================================================================
# CELDA 14: Tabla Resumen Final (3 modelos)
# =============================================================================
print("\n" + "=" * 75)
print("RESUMEN FINAL DEL EXPERIMENTO — 3 MODELOS")
print("=" * 75)

models = [metrics_b, metrics_i, metrics_o]
print(f"Dataset: {DATA_PATH}")
print(f"  Registros: {len(df):,} | Anomalías reales: {int(y.sum())} ({y.mean()*100:.1f}%)")
print(f"\n{'Métrica':<28} {'Baseline':>12} {'Mejorado':>12} {'Óptimo':>12}")
print("-" * 68)
for key, label in [('precision','Precision'), ('recall','Recall'),
                    ('f1','F1-Score'), ('roc_auc','ROC-AUC')]:
    print(f"  {label:<26} {models[0][key]:>12.4f} {models[1][key]:>12.4f} {models[2][key]:>12.4f}")
print("-" * 68)
for key, label in [('tp','TP'), ('fp','FP (falsas alarmas)'),
                    ('fn','FN (no detectadas)')]:
    print(f"  {label:<26} {models[0][key]:>12} {models[1][key]:>12} {models[2][key]:>12}")
print("-" * 68)
print(f"  {'Anomalías detectadas':<26} {models[0]['tp']+models[0]['fp']:>12} {models[1]['tp']+models[1]['fp']:>12} {models[2]['tp']+models[2]['fp']:>12}")
print(f"\n📌 Conclusión sobre el dataset:")
print(f"   El AUC~0.55 indica que las anomalías se SUPERPONEN con los datos normales")
print(f"   en el espacio de features (Temperature, Humidity, Battery_Level).")
print(f"   Este es un dataset intencionalmente 'challenging' (difícil).")
print(f"   Isolation Forest funciona mejor cuando las anomalías son claramente")
print(f"   distintas en alguna dimensión del espacio de features.")
print()
print("✅ Script completado exitosamente.")
print(f"   Todas las figuras guardadas en: {OUTPUT_DIR}/")
