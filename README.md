# Isolation Forest — Detección de Anomalías en Sensores IoT

Caso práctico del algoritmo **Isolation Forest** aplicado a datos de sensores IoT (temperatura, humedad y nivel de batería).  
Desarrollado para el **Seminario: Algoritmos de Machine Learning** — Tema 9.

---

## Estructura del proyecto

```
iot-temperature-anomaly-detection/
├── data/
│   └── synthetic_iot_dataset_challenging.csv   # Dataset principal (3,000 registros)
├── notebook/
│   ├── caso_practico.ipynb                      # Jupyter Notebook narrado (12 celdas)
│   ├── caso_practico.py                         # Script Python equivalente
│   └── informe_caso_practico.md                 # Informe técnico completo
├── outputs/
│   └── notebook/                                # Figuras generadas (9 graficas .png)
├── src/
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_improved.py
│   ├── evaluate.py
│   └── visualize.py
├── main_baseline.py
├── main_improved.py
├── requirements.txt
└── README.md
```

---

## Dataset

**Archivo:** `data/synthetic_iot_dataset_challenging.csv`

| Columna | Descripción |
|---|---|
| `Device_ID` | Identificador del sensor (DHT11_A ... D) |
| `Temperature` | Lectura de temperatura (estandarizada) |
| `Humidity` | Lectura de humedad (estandarizada) |
| `Battery_Level` | Nivel de batería (estandarizado) |
| `Anomaly` | Ground truth: 0 = normal, 1 = anomalía |

- **3,000 registros** — 4 sensores DHT11
- **17.4% anomalías** (522 registros)

---

## Instalación

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### Caso práctico (notebook)

```bash
# Ejecutar el script completo
python3 notebook/caso_practico.py

# O abrir el notebook en Jupyter
jupyter notebook notebook/caso_practico.ipynb
```

Las 9 figuras se guardan automáticamente en `outputs/notebook/`.

### Modelos baseline e improved

```bash
python3 main_baseline.py
python3 main_improved.py
```

---

## Resultados

| Modelo | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline (contamination='auto') | 0.1649 | 0.1743 | 0.1695 | 0.5519 |
| Mejorado (contamination=0.05) | 0.1333 | 0.0383 | 0.0595 | 0.5579 |
| Optimo (contamination=0.174) | 0.1724 | 0.1724 | 0.1724 | 0.5579 |

> El dataset es intencionalmente desafiante: las anomalias se superponen con datos normales en el espacio de features, lo que resulta en un ROC-AUC cercano a 0.55. Esto demuestra las limitaciones del algoritmo cuando no hay separacion clara entre clases.

---

## Referencias

- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. IEEE ICDM.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
