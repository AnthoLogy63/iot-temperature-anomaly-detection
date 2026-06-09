# Isolation Forest IoT

Detección de anomalías en datos de sensores IoT usando Isolation Forest.

## Estructura del proyecto

```
isolation-forest-iot/
├── data/               # Datasets de entrada
│   └── iot_sensor.csv
├── outputs/            # Resultados generados
│   ├── baseline/       # Resultados del modelo baseline
│   └── improved/       # Resultados del modelo mejorado
├── src/                # Código fuente
│   ├── preprocess.py   # Preprocesamiento de datos
│   ├── train_baseline.py
│   ├── train_improved.py
│   ├── evaluate.py     # Evaluación de resultados
│   └── visualize.py    # Visualización de anomalías
├── main_baseline.py    # Entry point modelo baseline
├── main_improved.py    # Entry point modelo mejorado
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main_baseline.py
python main_improved.py
```
