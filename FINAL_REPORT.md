# NextLap · Informe final y repositorio

**IIT414W · Capstone NextLap · Equipo G07 Castro–Bufadel**  
**Integrantes:** Valentín Castro · Camilo Bufadel  
**Repositorio:** https://github.com/Lilvecp/capstone  
**Commit evaluado (SHA):** `COMMIT_SHA` (tag `Final`)  
**Fecha:** 22-11-2026

**Resumen.** Predecimos al final de la vuelta t la duración de la vuelta t+1 del mismo piloto (`next_lap_s`, s). La hipótesis fue que el **tráfico** (distancia al auto de adelante en la línea) agrega información a la referencia OLS de 6 inputs. Probamos tres representaciones con su control pareado: gap con tope 5 s (Hito 2), cambio del gap entre t−1 y t (pedido en el feedback del Hito 2) y gap solo a autos **en la misma vuelta** (prioridad de la Demo). Ninguna cumple los criterios fijados antes de evaluar. **Decisión final: retener la referencia OLS de 6 inputs (E0).** Es un resultado negativo acotado a OLS, población `clean_pair` y R8–R9.

---

## 1. Validez de la evaluación  — [Valentín]

### 1a. Split y ajuste (C1a)
- **Target y momento:** `next_lap_s` (s) predicho al final de la vuelta t (`end_time_s` de t).
- **Split por carrera:** entrenamiento R1–R7 (5.772 pares clean), validación de desarrollo R8–R9 (1.808: R8 Austria 1.066, R9 Gran Bretaña 742). Sin test: el paquete no lo incluye y no se usó. Seed 414.
- **Ajuste solo con train:** todos los modelos son `make_model('OLS')` del helper (imputación por mediana + `StandardScaler` + one-hot dentro del `Pipeline`); `.fit` solo con R1–R7, validación solo pasa por `.predict`. Evidencia en la corrida: checks K0b/K4 (`salidas_hito2/checks.csv`) y F0b/F7/F7b (`salidas_final/checks_final.csv`): la media del scaler de cada feature nueva es la de train (p. ej. `gap_delta_s`: 0,05119 train vs 0,04942 train+val).

### 1b. Disponibilidad de los inputs nuevos (C1b)

| Input | Fuente y momento | Última observación usada | Faltantes / historial corto |
|---|---|---|---|
| `gap_ahead_cap_s` (E1) | `base_NextLap/data/race_2026_r0X_laps_v2.csv` (SHA-256 vs `source_manifest_v2.json`). `end_time_s(t)` − `end_time_s` del auto (otro piloto) que cruzó justo antes; tope 5 s | Cruce del otro auto con `end_time_s ≤ end_time_s(t)` (K1: máx. diferencia 0,000 s) | Sin auto antes → 5 s. 0 filas en train y validación |
| `gap_delta_s` (E3a) | `gap_cap(t) − gap_cap(t−1)`, t−1 = vuelta anterior del mismo piloto, carrera y stint | Fin de t−1 y cruce de su auto de adelante, ambos antes del fin de t (F1: máx. `end_time(t−1) − end_time(t)` = −70,4 s) | Sin t−1 en el stint → 0. 1 fila en train, 0 en validación |
| `gap_samelap_cap_s` (E4) | Igual que E1, pero el auto de adelante debe completar **el mismo número de vuelta** que t (ignora doblados) | Cruce del otro auto en esa vuelta, antes del fin de t (F1b) | Nadie completó antes esa vuelta (va primero) → 5 s. 323 filas en train, 92 en validación |

Todas se calculan sobre la **línea de tiempo original completa** (incluye vueltas de pit/SC/no precisas de los demás) **antes** del filtro retrospectivo `clean_pair`, y se unen a los pares por `pair_id` (K3b: 503 pares tendrían otro gap si se filtrara primero). **Población evaluada:** `clean_pair` es una población condicional retrospectiva (usa condiciones de t+1: vuelta precisa, sin pit, bandera verde, mismo stint). No es evidencia de desempeño en vivo ni en carrera completa, y **no excluye tráfico**: el 18,1 % de los pares de validación tiene un auto a < 1 s (corrección del feedback C1.3 del Hito 1).

### 1c. Mismos IDs de validación (C1c)
Todas las versiones se evalúan en las **mismas 1.808 filas**: `pair_id` únicos (K0d/F0d), join uno a uno sin pérdidas ni duplicados (K2a/F2a: 1.808 → 1.808), hash del set = `f453a98dc906904c…` en Hito 1, Hito 2 y final (K2b/F2b), y mismo orden y mismas predicciones de E0 que en el Hito 2 (F6). Cobertura 100 %: todas las filas tienen predicción finita en todas las versiones (K8/F8).

## 2. Comparación y decisión  — [Valentín]

### 2a. Corridas ejecutadas (C2a)

| Versión | Modelo e inputs | Rol | n | MAE (s) | Salida |
|---|---|---|---|---|---|
| **E0** | OLS · 6 inputs publicados | Referencia **y** control pareado de E1, E3a y E4 | 1.808 | **0,38671** | `salidas_hito2/validation_predictions.csv` · `pred_E0_reference_ols` |
| E1 | E0 + `gap_ahead_cap_s` | Candidato Hito 2 | 1.808 | 0,38705 | idem · `pred_E1_candidate` |
| E2 | E0 + gap permutado dentro de carrera (seed 414) | Placebo de E1 | 1.808 | 0,38678 | idem · `pred_E2_placebo` |
| S1 | E0 + gap sin autos en pit in/out | Sensibilidad de E1 | 1.808 | 0,38731 | idem · `pred_S1_sensitivity_nopit` |
| E3a | E0 + `gap_delta_s` | Candidato del feedback Hito 2 | 1.808 | 0,38692 | `salidas_final/validation_predictions_final.csv` · `pred_E3a_delta` |
| E4 | E0 + `gap_samelap_cap_s` | Candidato de la prioridad de la Demo | 1.808 | 0,38670 | idem · `pred_E4_samelap_gap` |

Cada candidato es **el mismo OLS** con una columna más; su control pareado es E0 reajustado sin esa columna (mismo `Pipeline`, sin hiperparámetros, mismas filas, misma métrica). Los modelos se reajustan en cada corrida: E0 y E1 se reproducen en el script final con la misma MAE del Hito 2 (F5, |dif| < 1e-9 s).

### 2b. Diferencias medidas (C2b)

Control − candidato (positivo = el candidato tiene menor MAE). IC 95 % bootstrap por bloques piloto-carrera (43 bloques, 2.000 remuestreos, seed 414).

| Contraste | n | MAE control (s) | MAE candidato (s) | Control − candidato | IC 95 % | Fuente |
|---|---|---|---|---|---|---|
| E0 − E1 (gap, Hito 2) | 1.808 | 0,38671 | 0,38705 | **−0,343 ms** | [−2,71 ; +2,09] ms | `salidas_hito2/run.json` |
| E0 − E3a (cambio del gap) | 1.808 | 0,38671 | 0,38692 | **−0,208 ms** | [−1,32 ; +0,88] ms | `salidas_final/run_final.json` |
| E0 − E4 (gap misma vuelta) | 1.808 | 0,38671 | 0,38670 | **+0,006 ms** | [−0,39 ; +0,40] ms | idem |
| E1 − E4 (efecto de restringir a la misma vuelta) | 1.808 | 0,38705 | 0,38670 | +0,348 ms | [−1,77 ; +2,37] ms | idem |

Las diferencias se recalculan desde las predicciones por fila y el global coincide con R8/R9 ponderado por n (K9, F9).

**Placebo acotado (corrección prioritaria del Hito 2).** En el Hito 2 afirmamos, con **una** permutación, que "la columna nueva no aporta más que ruido". Esa lectura solo vale para esa permutación. Repetimos el placebo con 200 permutaciones dentro de carrera (semillas 414–613; la 414 reproduce la del Hito 2, F10): MAE placebo media 0,38689 s, rango 2,5–97,5 % [0,38652 ; 0,38762] s. La permutación del Hito 2 cae en el percentil 48,5. **E1 (0,38705 s) es peor que el 77 % de los placebos**, y el 26,5 % de los placebos queda por debajo de E0. Es decir: con esta muestra, el gap real no se distingue de una columna con la misma distribución sin vínculo con la fila (`salidas_final/placebo_E1_permutations.csv`).

### 2c. Decisión (C2c)
**Se retiene E0 y se descartan las tres representaciones del tráfico.** Razones, en el orden del pre-registro:
1. **E1** empeora a E0 (−0,343 ms), con IC que incluye 0, y no supera a la mayoría de los placebos.
2. **E3a** (cambio del gap) empeora a E0 globalmente (−0,208 ms) y en **ambas** carreras (R8 −0,26 ms, R9 −0,13 ms), y no mejora el tramo < 1 s (−0,65 ms). Ya en la selección dentro de R1–R7 (LORO) quedaba peor que E0 (0,50919 vs 0,50839 s).
3. **E4** (misma vuelta) iguala a E0: +0,006 ms global, IC [−0,39 ; +0,40] ms centrado en 0, signos opuestos por carrera (R8 −0,37 ms, R9 +0,55 ms) y peor en el tramo < 1 s (−1,13 ms). No cumple el criterio "apoya" (mejora global **y** en R8 **y** en R9 **y** en < 1 s). Restringir a la misma vuelta sí elimina el daño de E1 (+0,348 ms frente a E1), pero no agrega información sobre E0.

**Alcance:** OLS lineal, representaciones con tope 5 s, población `clean_pair`, dos carreras de validación. No prueba que el tráfico no importe; prueba que **estas** representaciones no mejoran la referencia aquí. Por lo mismo, **no nominamos candidato al bono** de mejora: la ganancia de E4 (+0,006 ms) está dentro del ruido y no cumple nuestros propios criterios.

## 3. Errores por contexto  — [Camilo · PENDIENTE]

> Insumos listos: `salidas_hito2/control_vs_candidate_by_race.csv` (E0 vs E1), `salidas_final/control_vs_candidate_by_race_final.csv` (E0 vs E3a, E0 vs E4, E1 vs E4), `validation_predictions_final.csv`, `mae_by_gap_bin_final.csv`, `mae_by_samelap_gap_bin_E4.csv`, `ahead_car_other_lap_by_round.csv`.
> Falta: (3a) tabla R8/R9 con n, MAE y control − candidato para el candidato final que discutan (sugerencia: E4 vs E0 y E1 vs E0) y su relación con el global ponderado por n; (3b) caso inspeccionado (`2026-R8-3-L10` u otro) con observado, predicción y |error| verificado; (3c) limitación observada + siguiente check concreto. Dato útil: en el 32,9 % de los pares de validación el "auto de adelante" del Hito 2 iba en otra vuelta (Canadá 48,7 % en train).

## 4. Interpretación del modelo (SHAP)  — [Camilo · PENDIENTE]

> Insumos: `hito2_shap.py`, `salidas_hito2/shap_*`. Modelo retenido = E0.
> Falta: (4a) SHAP local de E0 con base + contribuciones = predicción y el |error| aparte; **corregir** la lectura del Hito 2 según el feedback: en Austria el SHAP del gap en E1 (−50,9 ms) se mide respecto de la base de E1; el cambio E1 − E0 de la predicción es −44,5 ms y también refleja el reajuste del modelo (no son la misma cantidad). (4b) SHAP global: muestra, explainer, fondo de train y alcance.

## 5. Reproducibilidad y decisiones  — [Valentín]

### 5a. Repositorio ejecutable (C5a)
| Qué | Dónde |
|---|---|
| URL y commit | https://github.com/Lilvecp/capstone · SHA en la portada (tag `Final`) |
| Datos | `base_NextLap/` = release `IIT414W_Capstone_NextLap_StudentData_v1.zip` sin cambios; SHA-256 en `README.md`, verificados en cada corrida (K0a/F0a) y snapshots contra `source_manifest_v2.json` |
| Entorno | `requirements.txt` (numpy, pandas, scikit-learn, joblib, shap); versiones exactas en `salidas_hito2/run.json`, `salidas_hito2/shap_run.json` y `salidas_final/run_final.json` → `environment` |
| Seed | `RANDOM_SEED = 414` (placebo, bootstrap, fondo SHAP); OLS es determinista |
| Comandos (desde la raíz, en orden) | `python hito1_reference.py` → `python hito2_experiment.py` → `python hito2_shap.py` → `python final_experiment.py` |
| Salidas | `salidas_hito1/`, `salidas_hito2/`, `salidas_final/` (tabla código → salida en `README.md`) |

### 5b. Registro de decisiones (C5b)
`DECISION_LOG.md` enlaza cada trial con su configuración y resultado, incluidos los no exitosos: `salidas_hito2/trials_log.csv` (7 corridas) y `salidas_final/trials_log_final.csv` (5 LORO en train + 4 en validación). Decisiones con evidencia por integrante:
- **Valentín:** usar E0 como control pareado de todos los candidatos (C2a); descartar E1 (C2c, IC y placebo); pre-registrar y ejecutar E3a y E4 con selección en R1–R7; acotar el placebo con 200 permutaciones; no nominar al bono.
- **Camilo:** elegir explicar E0 con SHAP por ser el modelo retenido; el caso `2026-R8-3-L10` y el análisis del desplazamiento por tramo de gap que llevó al siguiente check (cambio del gap); *(completar con su decisión del informe final)*.

### 5c. Uso de IA (C5c)
| Quién | Uso real | Qué se adoptó | Verificación y resultado |
|---|---|---|---|
| Valentín | Claude: leer el enunciado y el feedback en Canvas, proponer el reparto, escribir `hito2_experiment.py`, `final_experiment.py` y el borrador de §1, §2, §5 y §6. Prompt representativo: «ahora viene esto así que divídelo en dos igual y trabaja» | El reparto, la estructura de ambos scripts, los checks K/F, el bootstrap por bloques, el placebo con 200 permutaciones. Las representaciones E3a y E4 salen del feedback del profe y del comentario de la Demo | 23 checks OK en `final_experiment.py`; E0/E1 reproducen el Hito 2 (F5) y la semilla 414 reproduce el placebo del Hito 2 (F10); recalculé el global como promedio ponderado de R8 y R9 (F9) y contrasté cada cifra de §2 con `salidas_final/*.csv` |
| Camilo | *(completar)* | | |

## 6. Revisión tras feedback  — [Valentín]

### 6a. Prioridad y respuesta (C6a)
| Fuente | Prioridad | Respuesta |
|---|---|---|
| **Demo (20-11)** | El "auto de adelante" puede ir en otra vuelta (doblado o que nos dobló), sobre todo en circuitos cortos; hay que limitar el gap a la misma vuelta | **Cambio:** nueva representación E4 (`gap_samelap_cap_s`), pre-registrada antes de evaluar (`DECISION_LOG.md`) |
| **Feedback Hito 2 · comprobación** | Fijar antes de evaluar una representación del cambio del gap t−1→t, compararla con E0 en los mismos IDs con selección en R1–R7, ver el tramo < 1 s y reportar MAE total y por carrera | **Cambio:** E3a/E3b pre-registradas; LORO eligió E3a; comparada con E0 |
| **Feedback Hito 2 · corrección prioritaria** | Acotar la lectura del placebo a la permutación observada | **Cambio:** placebo con 200 permutaciones; lectura reescrita en §2b. La corrección SHAP (−50,9 vs −44,5 ms) va en §4 |

Las dos primeras apuntan al mismo problema: el gap en la línea no distingue situaciones de tráfico distintas. Tras ambas revisiones **mantenemos la decisión** de retener E0.

### 6b. Verificación (C6b)

| Check | Resultado | Evidencia |
|---|---|---|
| ¿El problema de la Demo existe? (sin target) | Sí: el auto usado por E1 iba en otra vuelta en el **29,5 %** de los pares de train y el **32,9 %** de validación (Canadá 48,7 %, Japón 8,0 %) | `salidas_final/ahead_car_other_lap_by_round.csv`, `feature_coverage_final.csv` |
| E4 vs E0 (misma vuelta) | +0,006 ms global, IC [−0,39 ; +0,40] ms; R8 −0,37 ms (n = 1.066), R9 +0,55 ms (n = 742); tramo < 1 s (misma vuelta, n = 274) −1,21 ms → no apoya | `control_vs_candidate_by_race_final.csv`, `mae_by_samelap_gap_bin_E4.csv` |
| E4 vs E1 | +0,348 ms: restringir a la misma vuelta quita el daño de E1, sin superar a E0 | `control_vs_candidate_by_race_final.csv` |
| Selección del cambio del gap en R1–R7 | LORO: E0 0,50839 · E3a 0,50919 · E3b 0,50939 s → E3a (E3b no la supera por > 0,005 s) | `selection_loro_train_final.csv` |
| E3a vs E0, total y por carrera | Global −0,208 ms (IC [−1,32 ; +0,88]); R8 −0,26 ms (n = 1.066); R9 −0,13 ms (n = 742) | `control_vs_candidate_by_race_final.csv`, `run_final.json` |
| E3a en el tramo < 1 s | MAE 0,45220 vs E0 0,45155 s (n = 328): −0,65 ms → **no** reduce el error donde había tráfico | `mae_by_gap_bin_final.csv` |
| Placebo acotado | Permutación del Hito 2 en el percentil 48,5 de 200; E1 peor que el 77 % de los placebos | `placebo_E1_permutations.csv`, `run_final.json` |

*Transparencia:* el pre-registro de E3a se escribió a las 14:14 y el script se ejecutó a las 14:16, antes de que el pre-registro quedara en un commit; no se modificó nada después (`DECISION_LOG.md`, nota de transparencia).

## 7. Contribuciones

- **Valentín Castro:** pipeline de evaluación y experimentos (`hito1_reference.py`, `hito2_experiment.py`, `final_experiment.py`), §1, §2, §5, §6, README y empaquetado.
- **Camilo Bufadel:** hipótesis de tráfico y plan (Hito 1), errores por contexto y SHAP (`hito2_shap.py`), §3, §4 *(completar con lo del informe final)*.
