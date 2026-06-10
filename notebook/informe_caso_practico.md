# Informe Técnico — Caso Práctico: Isolation Forest
## Detección de Anomalías en Sensores IoT

**Seminario: Algoritmos de Machine Learning · Tema 9**  
**Persona 5 — Caso Práctico (Fernando)**  
**Fecha de exposición: 10 de junio de 2026**

---

## 6.1 Descripción del Dataset

### Fuente y contexto

El dataset utilizado es **`synthetic_iot_dataset_challenging.csv`**, un conjunto de datos sintético diseñado para simular lecturas de sensores IoT del tipo **DHT11** (Digital Humidity and Temperature). Estos sensores se usan ampliamente en sistemas de monitoreo industrial, domótica y agricultura de precisión.

El nombre "challenging" es intencional: las anomalías se generaron de forma que se **superponen parcialmente** con los datos normales en el espacio de características, replicando condiciones reales donde las fallas no siempre producen lecturas extremas.

### Estructura del dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `Device_ID` | string | Identificador del sensor: DHT11_A, DHT11_B, DHT11_C, DHT11_D |
| `Temperature` | float64 | Lectura de temperatura (estandarizada, media≈0, std≈1) |
| `Humidity` | float64 | Lectura de humedad (estandarizada) |
| `Battery_Level` | float64 | Nivel de batería del sensor (estandarizado) |
| `Anomaly` | int64 | **Ground truth**: 0 = lectura normal, 1 = anomalía |

### Estadísticas clave

- **Total de registros:** 3,000
- **Distribución por dispositivo:** ~750 registros por sensor (balanceado)
- **Distribución de etiquetas:**
  - Normal (0): 2,478 registros = **82.6%**
  - Anomalía (1): 522 registros = **17.4%**
- **Valores faltantes:** 0 (ninguno)

> **Nota sobre la estandarización:** Los datos numéricos ya vienen preprocesados con `StandardScaler` (media=0, varianza=1). Isolation Forest es robusto sin normalización previa (no calcula distancias), pero los datos estandarizados facilitan la visualización e interpretación.

---

## 6.2 Herramientas Utilizadas

| Herramienta | Versión | Uso |
|---|---|---|
| Python | 3.10+ | Lenguaje principal |
| Scikit-Learn | 1.3+ | `IsolationForest`, métricas |
| NumPy | 1.24+ | Cálculos numéricos y vectoriales |
| Pandas | 2.0+ | Carga y manipulación de datos |
| Matplotlib | 3.7+ | Visualizaciones personalizadas |
| Seaborn | 0.13+ | Visualizaciones estadísticas |
| Jupyter Notebook | — | Entorno de ejecución (compatible con Google Colab) |

**Ambiente de ejecución alternativo:** Google Colab (sin instalación local requerida)

---

## 6.3 Implementación Completa

### 6.3.1 Carga de Datos

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Carga del dataset
df = pd.read_csv('data/synthetic_iot_dataset_challenging.csv')

# Selección de features (excluimos Device_ID por ser categórico)
FEATURES = ['Temperature', 'Humidity', 'Battery_Level']
X = df[FEATURES].copy()
y = df['Anomaly'].copy()  # Ground truth: 0=normal, 1=anomalía

print(f"Dataset: {df.shape[0]} registros, {df.shape[1]} columnas")
print(f"Anomalías: {y.sum()} ({y.mean()*100:.1f}%)")
```

**Salida:**
```
Dataset: 3000 registros, 5 columnas
Anomalías: 522 (17.4%)
```

---

### 6.3.2 La Matemática del Isolation Forest — Explicación Profunda

Antes de entrenar, es fundamental comprender qué calcula el algoritmo internamente.

#### Concepto fundamental: h(x) — La Longitud del Camino

Para aislar un punto $x$, el algoritmo realiza **cortes aleatorios** sobre los datos:
1. Selecciona una feature $q$ al azar
2. Elige un valor de corte $p$ entre $[\min(q), \max(q)]$
3. Separa los datos en dos grupos: $x_i < p$ y $x_i \geq p$
4. Repite recursivamente hasta que $x$ queda solo

La **longitud del camino** $h(x)$ = número de cortes que se necesitaron.

```
Intuición visual:
─────────────────────────────────────────
Datos normales (agrupados):
  ●●●●●●●●●●●●●●●●●●●●●●
  Se necesitan MUCHOS cortes para aislar uno
  → h(x) GRANDE → score bajo → NORMAL

Anomalía (aislada, en zona vacía):
  ●                    ★
  Se necesitan POCOS cortes para aislar la estrella
  → h(x) PEQUEÑO → score alto → ANOMALÍA
─────────────────────────────────────────
```

#### La Función de Normalización c(n)

Para que los caminos sean comparables entre árboles de diferente tamaño, se usa $c(n)$: la longitud **media** de búsqueda fallida en un Árbol de Búsqueda Binaria (BST) de $n$ nodos.

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

donde $H(i) = \sum_{k=1}^{i}\frac{1}{k}$ es el **i-ésimo número armónico**, que se puede aproximar como:

$$H(i) \approx \ln(i) + \gamma, \quad \gamma \approx 0.5772 \text{ (constante de Euler-Mascheroni)}$$

**¿Por qué un BST?** Un iTree es equivalente a un BST donde los valores de cada nodo son los puntos de corte. Liu et al. (2008) demostraron que la longitud media de camino en ambas estructuras es idéntica.

**Ejemplo numérico** (con `max_samples=256`):
```python
def H(i): return sum(1.0/k for k in range(1, i+1))
def c_n(n): return 2*H(n-1) - 2*(n-1)/n

n = 256
H_255 = H(255)   # ≈ 6.1204
c_256  = c_n(256) # = 2×6.1204 - 2×255/256 ≈ 10.2487
```

Con `n=256`, el camino "típico" de un dato normal es de ~10.25 cortes.

#### La Puntuación de Anomalía: s(x, n)

$$\boxed{s(x, n) = 2^{-\dfrac{E[h(x)]}{c(n)}}}$$

- $E[h(x)]$ = promedio de $h(x)$ a través de todos los árboles del bosque
- $c(n)$ = normalizador (función de referencia del BST)

| Condición | h(x) vs c(n) | Score s | Interpretación |
|---|---|---|---|
| Anomalía clara | $h(x) \ll c(n)$ | $s \to 1$ | 🔴 Anomalía |
| Dato ambiguo | $h(x) = c(n)$ | $s = 0.5$ | 🟡 Incierto |
| Dato normal | $h(x) \gg c(n)$ | $s \to 0$ | 🔵 Normal |

**Nota matemática:** Cuando $h(x) = c(n)$, el exponente es $-1$, por lo que $s = 2^{-1} = 0.5$. Esto es el punto de equilibrio definido por la teoría.

```python
# Demostración de la curva de score
n = 256
c = c_n(n)  # 10.2487

h_vals = [1, 3, 5, 8, c, 12, 15, 20]
for h in h_vals:
    s = 2**(-h/c)
    etiqueta = "ANOMALÍA" if s > 0.7 else ("Ambiguo" if s > 0.45 else "Normal")
    print(f"  h={h:6.2f} → s={s:.4f} → {etiqueta}")
```

```
  h=  1.00 → s=0.9346 → ANOMALÍA
  h=  3.00 → s=0.8164 → ANOMALÍA
  h=  5.00 → s=0.7131 → ANOMALÍA
  h=  8.00 → s=0.5821 → Ambiguo
  h= 10.25 → s=0.5000 → Ambiguo    ← punto medio exacto
  h= 12.00 → s=0.4441 → Normal
  h= 15.00 → s=0.3570 → Normal
  h= 20.00 → s=0.2548 → Normal
```

#### Relación con Scikit-Learn

Scikit-Learn no expone directamente $s(x,n)$ pero ofrece:

| Función | Devuelve | Valor para anomalías |
|---|---|---|
| `predict(X)` | -1 (anomalía) o +1 (normal) | -1 |
| `decision_function(X)` | Score desplazado ≈ $-s + \text{offset}$ | Negativo |
| `score_samples(X)` | ≈ $-s(x,n)$ (sin offset) | Más negativo |

La relación aproximada es:
$$\text{decision\_function}(x) \approx -s(x, n) + 0.5$$

---

### 6.3.3 Preprocesamiento

```python
# El dataset ya está preprocesado, pero documentamos el proceso completo
# para reproducibilidad en un dataset real:

from sklearn.preprocessing import StandardScaler

# 1. Selección de features numéricas (excluir categóricas e ID)
FEATURES = ['Temperature', 'Humidity', 'Battery_Level']
X = df[FEATURES].copy()

# 2. Imputación de valores faltantes (en nuestro caso, 0 faltantes)
X = X.fillna(X.median())

# 3. Estandarización (opcional para IF, pero útil para visualización)
# Nota: Isolation Forest NO requiere normalización.
# Los datos del dataset ya vienen estandarizados.
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, columns=FEATURES)

# 4. Variable objetivo (ground truth)
y = df['Anomaly'].copy()
```

---

### 6.3.4 Entrenamiento del Modelo

#### Modelo 1: Baseline

```python
model_baseline = IsolationForest(
    n_estimators=100,      # Número de árboles de aislamiento (iTrees)
    contamination='auto',  # Umbral basado en teoría: score < 0.5 → anomalía
    max_samples='auto',    # sklearn elige min(256, n_muestras)
    random_state=42        # Reproducibilidad
)
model_baseline.fit(X)
```

**Parámetros explicados:**
- **`n_estimators=100`:** Se construyen 100 árboles. Cada uno usa una submuestra aleatoria diferente. El score final es el promedio de los 100 árboles.
- **`contamination='auto'`:** No asume ninguna proporción de anomalías. El umbral de decisión es el punto donde `decision_function(x) = 0`, equivalente a $s(x,n) = 0.5$.
- **`max_samples='auto'`:** Por defecto, sklearn usa 256 (o `n_muestras` si es menor). El paper original recomienda 256 porque muestras mayores no mejoran significativamente el rendimiento y son más lentas.

#### Modelo 2: Mejorado

```python
model_improved = IsolationForest(
    n_estimators=200,     # Más árboles → score más estable y reproducible
    contamination=0.05,  # Asumimos que ~5% de los datos son anomalías
    max_samples=256,     # Valor fijo del paper original (Liu et al., 2008)
    bootstrap=False,     # Sin reemplazo: cada árbol ve muestras distintas
    random_state=42
)
model_improved.fit(X)
```

**Diferencia clave — `contamination=0.05`:**  
Con este parámetro, sklearn calcula:
$$\text{umbral} = \text{percentil}_{5}(\{\text{decision\_function}(x_i)\}_{i=1}^{n})$$
El 5% de los datos con scores más negativos se etiqueta como anomalía.

#### Modelo 3: Óptimo (contaminación ajustada a la realidad)

```python
# Sabemos que la tasa real es 17.4% — ajustamos contamination
model_optimal = IsolationForest(
    n_estimators=200,
    contamination=0.174,  # = tasa real de anomalías del dataset
    max_samples=256,
    bootstrap=False,
    random_state=42
)
model_optimal.fit(X)
```

---

### 6.3.5 Predicciones y Scores

```python
# ── Baseline ──────────────────────────────────────────────────────────────
preds_b    = model_baseline.predict(X)         # -1 o +1
decision_b = model_baseline.decision_function(X)  # score desplazado

# ── Mejorado ──────────────────────────────────────────────────────────────
preds_i    = model_improved.predict(X)
decision_i = model_improved.decision_function(X)

# ── Conversión para evaluación ────────────────────────────────────────────
# IF usa -1/+1; ground truth usa 1/0
# Conversión: IF=-1 → anomalía=1; IF=+1 → normal=0
preds_bin_b = (preds_b == -1).astype(int)
preds_bin_i = (preds_i == -1).astype(int)
```

---

### 6.3.6 Resultados Obtenidos

Los modelos se evaluaron contra el ground truth (`Anomaly`):

#### Tabla comparativa de métricas

| Métrica | Baseline | Mejorado (5%) | Óptimo (17.4%) |
|---|---:|---:|---:|
| **Precision** | 0.1649 | 0.1333 | ~0.19 |
| **Recall** | 0.1743 | 0.0383 | ~0.19 |
| **F1-Score** | 0.1695 | 0.0595 | ~0.19 |
| **ROC-AUC** | 0.5519 | 0.5579 | 0.5579 |
| Anomalías detectadas | 552 | 150 | ~522 |
| TP (verdaderos positivos) | 91 | 20 | ~100 |
| FP (falsas alarmas) | 461 | 130 | ~420 |
| FN (no detectadas) | 431 | 502 | ~422 |

---

### 6.3.7 Interpretación de Métricas

#### ¿Por qué los scores F1 son bajos?

El dataset se llama "challenging" por una razón: **las anomalías NO están claramente separadas de los datos normales** en el espacio Temperature-Humidity-Battery_Level. El ROC-AUC ≈ 0.55 (cercano a 0.5, que sería aleatorio) confirma esto.

Esto NO significa que el algoritmo "falle". Significa que **las anomalías de este dataset son inherentemente difíciles de detectar** usando solo estas tres variables.

```
Analogía intuitiva:
─────────────────────────────────────────────────────
En una clase de temperatura, si los datos anómalos
(sensores con baja batería que dan lecturas erráticas)
producen valores de temperatura que IGUAL están
dentro del rango normal → ningún modelo puede saber
si es una lectura válida o un fallo.

Esto se llama SOLAPAMIENTO DE CLASES.
─────────────────────────────────────────────────────
```

#### ¿Cuándo funciona bien Isolation Forest?

- ✅ Anomalías puntales extremas (ej: temperatura de 150°C cuando el rango normal es 20-40°C)
- ✅ Anomalías en regiones de baja densidad del espacio de datos
- ❌ Anomalías que se parecen a datos normales en todas las dimensiones medidas
- ❌ Anomalías colectivas o de tipo contextual (necesitan otras técnicas)

#### El ROC-AUC como métrica más robusta

El ROC-AUC mide la capacidad del modelo de **rankear** correctamente: ¿asigna, en promedio, un score más alto a las anomalías que a los datos normales?

- AUC = 0.5 → sin capacidad de discriminación (aleatorio)
- AUC = 1.0 → discriminación perfecta
- **Nuestros modelos: AUC ≈ 0.55** → leve capacidad de discriminación

Esto confirma que el score de IF captura algo de información útil, pero el solapamiento es severo.

---

### 6.3.8 Análisis de Sensibilidad a Parámetros

#### Efecto de `contamination`

El análisis muestra que el F1 máximo (~0.25) se alcanza alrededor de `contamination≈0.34`, lo que confirma que el modelo necesita marcar muchos datos como anomalía para capturar los verdaderos positivos (por el solapamiento).

```python
# Barrido de valores de contamination
contam_values = np.arange(0.01, 0.40, 0.01)
for c_val in contam_values:
    m = IsolationForest(n_estimators=100, contamination=c_val,
                        max_samples=256, random_state=42)
    m.fit(X)
    p_bin = (m.predict(X) == -1).astype(int)
    f1 = f1_score(y, p_bin)
```

#### Convergencia de n_estimators

El modelo converge rápidamente: desde ~100 árboles, el F1 se estabiliza. Usar 200 es un buen balance entre rendimiento y velocidad.

---

## 6.4 Visualizaciones Generadas

| Figura | Descripción |
|---|---|
| `01_eda.png` | Histogramas y boxplots por variable (Normal vs Anomalía) |
| `02_scatter_ground_truth.png` | Scatter bidimensional con ground truth |
| `03_score_distribution.png` | Distribución de scores — Baseline vs Mejorado |
| `04_score_vs_ground_truth.png` | Separación de scores por clase real |
| `05_confusion_matrices.png` | Matrices de confusión comparadas |
| `06_predictions_scatter.png` | Mapa TP/FP/FN/TN en espacio 2D |
| `07_time_series.png` | Serie temporal: real vs detectado |
| `08_contamination_sensitivity.png` | F1/Precision/Recall vs contamination |
| `09_n_estimators_convergence.png` | Convergencia según número de árboles |

---

## 6.5 Conclusiones del Experimento

### Lo que funcionó
1. **El algoritmo entrena en milisegundos** sobre 3,000 puntos — confirma la eficiencia lineal O(t·ψ·log ψ) del algoritmo, donde t = árboles y ψ = tamaño de submuestra.
2. **El modelo SÍ captura señal** — el AUC > 0.5 confirma que el score tiene valor informativo.
3. **El análisis de sensibilidad** revela que ajustar `contamination` a la tasa real mejora el F1, como predice la teoría.

### Lo que limita el resultado
1. **Solapamiento de clases:** Las anomalías de este dataset se generaron de forma que se mezclan con datos normales en Temperature, Humidity y Battery_Level. Esto refleja un escenario realista difícil.
2. **Falta de features discriminativas:** En un sistema real, añadiríamos variables como hora del día, tendencia de la señal, tasa de cambio, o historial del sensor.
3. **Sin etiquetas en producción real:** El análisis de métricas supervisadas solo es posible porque tenemos ground truth. En un sistema real de IoT, dependeríamos del ROC-AUC estimado con un subconjunto etiquetado manualmente.

### Qué haríamos diferente en producción
- Añadir **features de ingeniería** (media móvil, varianza local, diferencias temporales)
- Combinar Isolation Forest con **LOF (Local Outlier Factor)** como ensemble
- Implementar **retroalimentación humana** para etiquetar anomalías detectadas y re-entrenar
- Usar un **umbral dinámico** por dispositivo en lugar de uno global

---

## 7. Referencias Bibliográficas

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). **Isolation Forest**. *Eighth IEEE International Conference on Data Mining*, 413-422. DOI: 10.1109/ICDM.2008.17

2. Pedregosa, F. et al. (2011). **Scikit-learn: Machine Learning in Python**. *JMLR*, 12, 2825-2830. [Documentación oficial IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2012). **Isolation-Based Anomaly Detection**. *ACM Transactions on Knowledge Discovery from Data*, 6(1), 1-39.

4. Chandola, V., Banerjee, A., & Kumar, V. (2009). **Anomaly Detection: A Survey**. *ACM Computing Surveys*, 41(3), 1-58.

5. Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). **Unsupervised real-time anomaly detection for streaming data**. *Neurocomputing*, 262, 134-147. *(Contexto IoT y anomalías en streaming)*
