# Hito 1 · Capstone NextLap (IIT414W)

Predecir, al final de la vuelta t, la duración de la vuelta t+1 del mismo piloto (`next_lap_s`, segundos).

Equipo: Castro–Bufadel (sin nombre formal) · Repo: https://github.com/Lilvecp/capstone · Commit entregado: ver tag `Hito1` y el texto de entrega

## Qué hay aquí

| Archivo | Contenido |
|---|---|
| `HITO1.md` | Encuadre, referencia reproducida, hipótesis y plan de experimento |
| `DECISION_LOG.md` | Decisiones, revisión tras feedback, uso de IA y contribuciones reales |
| `hito1_reference.py` | Reproduce mediana de train, persistencia y OLS, con 9 checks automáticos |
| `hito1_feasibility_traffic.py` | Factibilidad de la hipótesis (gap al auto de adelante): fuente, timestamp, cobertura y mismos `pair_id`. 7 checks. Sin modelos |
| `salidas_hito1/` | Salidas generadas por el script (ver abajo) |
| `base_NextLap/` | Datos y helper del curso (`IIT414W_Capstone_NextLap_StudentData_v1`, sin cambios) |
| `requirements.txt` | Dependencias |

## Datos

`base_NextLap/` es el release oficial `IIT414W_Capstone_NextLap_StudentData_v1.zip` (Canvas IIT414W → Archivos), sin modificaciones. Si se omite del repo, basta con descomprimir ese zip en `base_NextLap/`. El script verifica los SHA-256:

| Archivo | SHA-256 |
|---|---|
| `train_clean_v1.csv` | `43544f38cc9ef32b44886e6751e03522601265160086fe3a4956f04b87bd5c13` |
| `validation_clean_v1.csv` | `b19efb6a908af12a5d2c6bdde29aa40984295e62ce0f187a201b579278db7ee8` |
| `nextlap_reference_v1.py` | `f6c58925cd4a477bf3e1a5a2b3968e67df3a1cffabecee9e89701c3ac2d5f994` |

Los snapshots originales por carrera están en `base_NextLap/data/` (manifest: `source_manifest_v2.json`). No hay datos de test en el paquete y el script no los usa.

## Cómo ejecutar

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python hito1_reference.py
python hito1_feasibility_traffic.py
```

Tarda unos segundos. El primero imprime la tabla de MAE y termina con `9 checks OK`; el segundo imprime la cobertura y termina con `7 checks OK`. Requiere Python ≥ 3.11 (pandas ≥ 2.3). Si un check falla, se detiene con el motivo.

## Salidas (`salidas_hito1/`)

| Archivo | Contenido |
|---|---|
| `reference_metrics.csv` | MAE (s) y n por modelo, para toda la validación y por carrera |
| `paired_ols_vs_persistence.csv` | Diferencia pareada de MAE y % de filas donde OLS gana |
| `validation_reference_predictions.csv` | Predicción de cada modelo por `pair_id` (trazabilidad) |
| `ols_coefficients.csv` | Coeficientes del OLS sobre inputs estandarizados/one-hot |
| `checks.csv` | Resultado de los checks C1–C9 |
| `feasibility_traffic_*.csv/json` | Checks F1–F7, cobertura del gap en train/val y distribución por carrera de train |
| `reference_run.json` | Conteos, features, mediana de train, hash del set de `pair_id` de validación, SHA-256 y versiones del entorno |

## Atribución

`hito1_reference.py` reutiliza `make_model`, `NUMERIC`, `CATEGORICAL` y `TARGET` de `base_NextLap/nextlap_reference_v1.py` y sigue la lógica de `probar_referencia_v1.py` (Class and Hito 1 student package v2, IIT414W). Los checks, las salidas por carrera, la comparación pareada y la interpretación son del equipo.
