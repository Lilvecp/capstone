# Hito 1 · NextLap · Framing and experiment plan

Equipo: ________ · Integrantes: Valentín Castro, ________
Repo: ________ · Commit: ________ · Fecha: 12-11-2026

## 1. Problem and decision value  — [A · Valentín]

| Elemento | Definición |
|---|---|
| Target y unidad | `next_lap_s`: duración de la vuelta t+1 del **mismo piloto**, en **segundos**. |
| Momento de predicción | Al terminar la vuelta t (`end_time_s` de t), usando solo información observada hasta ese instante y dentro del mismo piloto/carrera/stint. |
| Lector previsto | Ingeniero de carrera / estratega en el muro de boxes que sigue el ritmo de su piloto durante un stint. |
| Decisión que informa | Fijar un **ritmo esperado para la próxima vuelta**. Si la vuelta real se desvía bastante más que el error típico (~0,4 s), es una señal para revisar la causa (degradación acelerada, tráfico, problema del auto) y decidir si hablar con el piloto o reevaluar la ventana de parada. |
| Qué NO establece | No dice **cuándo** conviene entrar a boxes, cuántas posiciones ganaría una estrategia ni el efecto causal del neumático: solo estima un tiempo de vuelta condicionado a que t+1 sea una vuelta limpia. |

**Límite de la población `clean_pair`.** `clean_pair` se define en retrospectiva con condiciones de t+1: vuelta precisa, sin entrada ni salida de boxes, no borrada, bandera verde (`track_status = 1`) y mismo stint/compuesto. En vivo no se sabe de antemano si t+1 será limpia. En las carreras de validación, solo el 82,4 % (R8) y el 71,8 % (R9) de los pares consecutivos observados son `clean_pair` (77 % en el total de R1–9: 7.580 de 9.849). Los MAE de este informe valen **solo para esa población condicional** y no son evidencia de desempeño en carrera completa ni en tiempo real. Las vueltas con safety car, pit o tráfico fuerte quedan fuera y ahí se espera un error mayor.

## 2. Reproduced reference  — [A · Valentín]

| Elemento | Valor |
|---|---|
| Datos | `IIT414W_Capstone_NextLap_StudentData_v1` (idéntico a `base_NextLap/` del package v2). SHA-256 verificados en el check C1: train `43544f38…`, validación `b19efb6a…`, helper `f6c58925…` |
| Split | Train: rondas 1–7 (Australia → Barcelona), **n = 5.772**. Validación: rondas 8–9 (Austria, Gran Bretaña), **n = 1.808**. Ningún `pair_id` compartido; el último evento de train (14-06) es anterior al primero de validación (28-06). Seed 414. Sin test. |
| Features | `current_lap_s`, `recent_mean_s`, `current_lap_number`, `current_tyre_life` (numéricas); `current_compound`, `current_team` (categóricas). Ninguna `next_*` (check C7). |
| Fit / transform | `make_model('OLS')` es un `Pipeline`: imputación por mediana + `StandardScaler` (numéricas) y moda + one-hot (categóricas), luego `LinearRegression`. `.fit` se llama **solo con train**; validación solo pasa por `.predict`. El check C8 confirma que las medias del scaler son las de train y no las de train+validación. La mediana de referencia (84,762 s) se calcula solo con train. |
| Comando | `python hito1_reference.py` → `salidas_hito1/` (métricas, predicciones por `pair_id`, checks, entorno). Reutiliza con atribución el helper del curso. |

**Resultados en validación, mismas 1.808 filas para los tres modelos (MAE en segundos):**

| Modelo | Toda la validación (n = 1.808) | R8 Austria (n = 1.066) | R9 Gran Bretaña (n = 742) |
|---|---|---|---|
| Mediana de train (84,762 s constante) | 11,376 | 11,716 | 10,888 |
| Persistencia (`next = current_lap_s`) | 0,409 | 0,401 | 0,422 |
| OLS referencia (6 inputs) | **0,387** | **0,374** | **0,406** |

Fuente: `salidas_hito1/reference_metrics.csv`. Hash del conjunto de `pair_id` de validación: `f453a98dc906904c…` (en `reference_run.json`), que sirve para comprobar en el Hito 2 que se comparan las mismas filas.

**Interpretación (OLS vs persistencia).** OLS baja el MAE en **0,023 s (−5,5 %)** frente a repetir la vuelta actual, y la mejora aparece en las dos carreras (−0,027 s en R8 y −0,016 s en R9). La ganancia es chica y poco uniforme: OLS tiene menor error absoluto en solo el **51,6 %** de las filas (49,6 % en R9), así que su ventaja viene de corregir menos casos con error grande y no de ser mejor vuelta a vuelta (`paired_ols_vs_persistence.csv`). Los coeficientes estandarizados más grandes son `recent_mean_s` (4,29) y `current_lap_s` (3,49), es decir, OLS es básicamente una persistencia suavizada. La mediana de train falla por ~11 s porque el nivel de tiempo de vuelta depende del circuito (73–109 s en train). Con dos carreras de validación, la diferencia es evidencia de selección, no una conclusión general.

*Nota de ejecución:* ver el entorno exacto (Python/numpy/pandas/sklearn) en `salidas_hito1/reference_run.json`. Los MAE coinciden con las salidas del instructor hasta ~1e-15 s.

## 3. Primary hypothesis  — [B · compañero/a]

- Feature / representación propuesta:
- Mecanismo esperado:
- Fuente:
- Disponibilidad en t (regla temporal y de agrupación):
- Regla para datos faltantes o historial corto:
- Plan B (si aplica):

## 4. Experiment plan  — [B · compañero/a]

- Qué queda fijo:
- Comparación referencia vs extensión:
- Ablación (extensión sin la info añadida):
- Qué apoyaría / contradiría la hipótesis:
- Uso de validación y protección del test:
- Búsqueda planificada (acotada) y registro de intentos:
- Checks previos (escritos antes de implementar):
- Robustez: MAE por carrera de validación con conteos

## 5. Traceability and next steps  — [ambos]

| Integrante | Decisiones/partes propias | Puede explicar |
|---|---|---|
| Valentín | | |
| ________ | | |

- Registro de IA: ver DECISION_LOG.md
- Revisión tras feedback:
- Responsables antes del Hito 2:
- Bloqueo conocido:
