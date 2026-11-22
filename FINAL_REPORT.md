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

## 3. Errores por contexto  — [Camilo]

Comando: `python final_errors.py` (después de `final_experiment.py`; lee sus predicciones, no reajusta modelos) → `salidas_final/errors_by_race_E1_E4.csv`, `errors_by_ahead_type.csv`, `errors_cases.csv`, `errors_checks.csv` (12 checks OK). Es un análisis **descriptivo** sobre validación: la decisión de §2 ya estaba tomada con los criterios pre-registrados.

### 3a. Por carrera (C3a)

Control − candidato (positivo = el candidato tiene menor MAE). Mismas 1.808 filas.

| Contraste | Carrera | n | MAE control (s) | MAE candidato (s) | Control − candidato | Filas donde gana el candidato |
|---|---|---|---|---|---|---|
| E0 − E1 (gap Hito 2) | R8 Austria | 1.066 | 0,37357 | 0,37549 | −1,92 ms | 47,5 % |
| | R9 Gran Bretaña | 742 | 0,40558 | 0,40365 | +1,93 ms | 50,8 % |
| | **Global** | 1.808 | 0,38671 | 0,38705 | **−0,34 ms** | 48,8 % |
| E0 − E4 (gap misma vuelta) | R8 Austria | 1.066 | 0,37357 | 0,37394 | −0,37 ms | 48,6 % |
| | R9 Gran Bretaña | 742 | 0,40558 | 0,40503 | +0,55 ms | 52,0 % |
| | **Global** | 1.808 | 0,38671 | 0,38670 | **+0,006 ms** | 50,0 % |

**Relación con el global.** El global es el promedio de R8 y R9 ponderado por n (check C1): para E4, (1.066 · −0,370 + 742 · +0,546) / 1.808 = +0,006 ms. En los dos candidatos el patrón es el mismo: **R8 favorece al control, R9 al candidato**, y el global queda cerca de 0 porque las dos carreras se compensan, no porque se parezcan. Restringir el gap a la misma vuelta (E4) reduce la magnitud de ambas diferencias (de ±1,9 ms a −0,4/+0,5 ms), pero no cambia el signo por carrera. Ningún candidato mejora en las dos carreras, y todas las diferencias son menores al 0,5 % de un MAE de ~0,39 s.

### 3b. Casos inspeccionados (C3b)

| | `2026-R8-3-L10` | `2026-R8-81-L69` |
|---|---|---|
| Contexto | Austria, piloto 3 (Red Bull), vuelta 10 → 11 | Austria, piloto 81 (McLaren), vuelta 69 → 70 |
| Auto de adelante (Hito 2) | **misma vuelta**, a 0,644 s | **otra vuelta**: Williams (23) completando la vuelta 67, a 0,237 s: el piloto lo está doblando |
| Gap misma vuelta (E4) | 0,644 s (mismo auto) | 15,863 s → tope 5 s |
| Observado t+1 (`y_true_s`) | 74,413 | 73,065 |
| E0: predicción → \|error\| | 72,708 → \|74,413 − 72,708\| = **1,705** | 72,385 → \|73,065 − 72,385\| = **0,680** |
| E1: predicción → \|error\| | 72,663 → **1,750** | 72,302 → **0,763** |
| E4: predicción → \|error\| | 72,699 → **1,714** | 72,385 → **0,680** |

Las restas coinciden con `abs_err_*` de `validation_predictions_final.csv` (check C5). **Lectura.** En `2026-R8-81-L69`, E1 trata a un doblado a 0,24 s como tráfico, baja la predicción 0,083 s y empeora el error. E4 ignora a ese auto (lo trata como aire libre) y vuelve exactamente al error de E0: es el problema señalado en la Demo, corregido. En `2026-R8-3-L10` el auto de adelante iba en la misma vuelta, así que E4 usa el mismo gap que E1: la predicción baja menos (−0,009 s frente a −0,044 s), porque E4 tiene otro coeficiente, pero sigue moviéndose hacia "más rápido" cuando la vuelta real fue 1,5 s más lenta. Ninguno de los tres modelos anticipa esa pérdida.

### 3c. Limitación observada y siguiente check (C3c)

Separando el "auto de adelante" del Hito 2 según su vuelta (`errors_by_ahead_type.csv`, n de validación):

| Auto de adelante (gap < 2 s) | n | MAE E0 (s) | E0 − E1 | E0 − E4 | t+1 más lenta que E0 |
|---|---|---|---|---|---|
| Misma vuelta | 596 | 0,373 | −6,2 ms | −0,9 ms | 58 % |
| Otra vuelta: el piloto **dobla** a un atrasado | 89 | 0,247 | −14,4 ms | +0,2 ms | 62 % |
| Otra vuelta: al piloto **lo dobla** uno adelantado | 82 | **0,960** | **+19,4 ms** | −1,3 ms | **24 %** |

Tres observaciones:
1. **En la misma vuelta (pelea real), ningún gap ayuda:** E1 y E4 empeoran a E0, sobre todo con gap < 1 s (E4 −1,2 ms, n = 273). En esa población el gap sigue sin distinguir "voy a adelantar" de "quedo atascado".
2. **Doblar a un atrasado** no es tráfico que cueste tiempo (MAE 0,25 s, el más bajo). E1 lo trataba como tráfico y empeoraba 14 ms; E4 lo corrige. Esto confirma el problema de la Demo.
3. **Ser doblado por un líder** es el contexto con el **mayor error de E0 (0,96 s, 2,6 veces el de la misma vuelta)**. Ahí la vuelta t+1 fue **más rápida** que lo predicho en el 76 % de los casos: la vuelta t probablemente incluyó ceder el paso, y E0, que repite el ritmo de t, sobreestima t+1. E1 bajaba la predicción y ganaba 19 ms en esas filas. **E4, al tratar a ese auto como "aire libre", pierde justamente esa información.** Por eso E4 elimina el daño de E1 sin superar a E0: corrige un caso y borra el otro.

**Limitación.** E4 trata igual dos situaciones opuestas en "otra vuelta" (doblar y ser doblado). El mayor error del modelo está concentrado en un grupo chico (82 filas, 4,5 % de validación) y en dos carreras, así que el patrón puede no repetirse en otras.

**Siguiente check (mismo protocolo).** Pre-registrar un indicador disponible en t: `being_lapped = 1` si el auto que cruzó justo antes, a menos de 2 s, va en una vuelta **mayor** que la del piloto (todo se calcula con `lap_number` y `end_time_s ≤ end_time_s(t)`). E5 = E0 + `gap_samelap_cap_s` + `being_lapped`, con selección LORO en R1–R7 y control pareado E4. Antes de mirar validación, verificar en R1–R7 que el grupo existe en cantidad suficiente y que E0 también comete ahí sus mayores errores. Si E5 no mejora en R8 **y** R9, se descarta igual.

## 4. Interpretación del modelo (SHAP)  — [Camilo]

Comando: `python hito2_shap.py` → `salidas_hito2/shap_local.csv`, `shap_global.csv`, `shap_checks.csv` (13 checks), `shap_run.json`. **Modelo explicado: E0**, el modelo retenido en §2c. E1 se reporta aparte solo para medir el peso del gap.

**Configuración.** `shap.LinearExplainer` (masker `Independent`) sobre la regresión ya preprocesada. **Fondo:** 1.000 filas de train R1–R7 (`sample(random_state=414)`), transformadas con el `prepare` ajustado solo con train. Las columnas one-hot se suman de vuelta a su input original (`current_compound`, `current_team`). Unidades: segundos de `next_lap_s`. Versiones: Python 3.12.13, sklearn 1.9.1, shap 0.54.1.

### 4a. Local: `2026-R8-3-L10` con E0 (C4a)

| Input | Valor | SHAP E0 (s) |
|---|---|---|
| **Base** (predicción media sobre el fondo R1–R7) | — | **87,2052** |
| `recent_mean_s` | 72,610 | −8,0462 |
| `current_lap_s` | 72,875 | −6,4044 |
| `current_tyre_life` | 10 | −0,0259 |
| `current_team` | Red Bull Racing | −0,0145 |
| `current_compound` | MEDIUM | −0,0055 |
| `current_lap_number` | 10 | −0,0009 |
| **Base + suma** | | **72,7078** = predicción de E0 ✓ (diferencia < 1e-6 s en las 1.808 filas, check S6) |
| *Aparte:* observado / \|error\| | | 74,413 / 1,705 |

**Lectura (asociación del modelo, no causalidad).** La base de 87,2 s mezcla circuitos más largos que Austria. Casi todo el descenso hasta 72,7 s (−14,45 s) viene de los dos inputs de ritmo: el modelo reconoce el **nivel de Austria y de este piloto**, no una dinámica propia de la vuelta. Neumático, equipo y compuesto mueven menos de 0,03 s cada uno. Ninguna contribución empuja hacia una vuelta más lenta, así que E0 no tenía con qué anticipar los +1,54 s observados. El \|error\| se reporta aparte porque SHAP explica la predicción, no el error.

**Corrección del Hito 2 (feedback).** En HITO2 §3 escribimos que la contribución del gap en E1 (−50,9 ms) "es lo que baja la predicción a 72,663". Eso mezclaba dos cantidades distintas:

| | E0 (s) | E1 (s) | E1 − E0 (ms) |
|---|---|---|---|
| Base | 87,2052 | 87,2040 | −1,1 |
| `current_lap_s` | −6,4044 | −6,5150 | −110,6 |
| `recent_mean_s` | −8,0462 | −7,9415 | +104,7 |
| `gap_ahead_cap_s` | — | −0,0509 | −50,9 |
| Resto (neumático, equipo, compuesto, n.º de vuelta) | −0,0468 | −0,0333 | +13,5 |
| **Predicción** | **72,7078** | **72,6633** | **−44,5** |

El −50,9 ms es la contribución del gap **dentro de E1, respecto de la base de E1** (el gap de esta fila, 0,644 s, frente al gap medio del fondo). El cambio de predicción E1 − E0 es −44,5 ms y **no es la misma cantidad**: al agregar la columna, el OLS se reajusta y cambian todos los coeficientes y la base. En esta fila, `current_lap_s` y `recent_mean_s` se redistribuyen ±105–111 ms entre sí y el resto aporta +13,5 ms. La lectura correcta es: *"en E1 el gap contribuye −50,9 ms respecto de la base de E1; la predicción de E1 queda 44,5 ms por debajo de la de E0, como efecto conjunto de la nueva columna y del reajuste"*.

### 4b. Global: media \|SHAP\| sobre las 1.808 filas de validación (C4b)

| Rank | Input (E0) | Media \|SHAP\| (s) | % | | Input (E1) | Media \|SHAP\| (s) |
|---|---|---|---|---|---|---|
| 1 | `recent_mean_s` | 6,509 | 54,7 % | | `recent_mean_s` | 6,424 |
| 2 | `current_lap_s` | 5,273 | 44,3 % | | `current_lap_s` | 5,364 |
| 3 | `current_compound` | 0,048 | 0,4 % | | `current_compound` | 0,048 |
| 4 | `current_team` | 0,033 | 0,3 % | | `gap_ahead_cap_s` | 0,037 |
| 5 | `current_tyre_life` | 0,028 | 0,2 % | | `current_team` | 0,033 |
| 6 | `current_lap_number` | 0,001 | 0,0 % | | `current_tyre_life` | 0,026 |
| 7 | | | | | `current_lap_number` | 0,007 |

**Muestra y agregación:** todas las filas de validación R8–R9 `clean_pair`, sin muestreo; media del valor absoluto por input original. **Lectura:** E0 se apoya en un 99 % en los dos inputs de ritmo; es una persistencia suavizada, coherente con los coeficientes del Hito 1. En E1 el gap mueve la predicción 0,037 s en promedio, un orden de magnitud menos que el MAE, y aun así no la mejora (§2).

**Alcance.**
1. Solo esta muestra y este fondo: como la base mezcla circuitos de R1–R7, gran parte de la \|SHAP\| de los inputs de ritmo es "nivel del circuito frente al fondo"; con un fondo del mismo circuito las magnitudes cambiarían.
2. SHAP describe en qué se apoya el **modelo**, no qué causa el tiempo de vuelta.
3. Una contribución mayor que 0 no prueba menor error.
4. El masker `Independent` reparte de forma convencional el peso entre `current_lap_s` y `recent_mean_s`, que están muy correlacionados. Por eso su reparto entre E0 y E1 (±105–111 ms en el caso) no tiene interpretación propia.

## 5. Reproducibilidad y decisiones  — [Valentín]

### 5a. Repositorio ejecutable (C5a)
| Qué | Dónde |
|---|---|
| URL y commit | https://github.com/Lilvecp/capstone · SHA en la portada (tag `Final`) |
| Datos | `base_NextLap/` = release `IIT414W_Capstone_NextLap_StudentData_v1.zip` sin cambios; SHA-256 en `README.md`, verificados en cada corrida (K0a/F0a) y snapshots contra `source_manifest_v2.json` |
| Entorno | `requirements.txt` (numpy, pandas, scikit-learn, joblib, shap); versiones exactas en `salidas_hito2/run.json`, `salidas_hito2/shap_run.json` y `salidas_final/run_final.json` → `environment` |
| Seed | `RANDOM_SEED = 414` (placebo, bootstrap, fondo SHAP); OLS es determinista |
| Comandos (desde la raíz, en orden) | `python hito1_reference.py` → `python hito2_experiment.py` → `python hito2_shap.py` → `python final_experiment.py` → `python final_errors.py` |
| Salidas | `salidas_hito1/`, `salidas_hito2/`, `salidas_final/` (tabla código → salida en `README.md`) |

### 5b. Registro de decisiones (C5b)
`DECISION_LOG.md` enlaza cada trial con su configuración y resultado, incluidos los no exitosos: `salidas_hito2/trials_log.csv` (7 corridas) y `salidas_final/trials_log_final.csv` (5 LORO en train + 4 en validación). Decisiones con evidencia por integrante:
- **Valentín:** usar E0 como control pareado de todos los candidatos (C2a); descartar E1 (C2c, IC y placebo); pre-registrar y ejecutar E3a y E4 con selección en R1–R7; acotar el placebo con 200 permutaciones; no nominar al bono.
- **Camilo:** elegir explicar E0 con SHAP por ser el modelo retenido; el caso `2026-R8-3-L10` y el análisis del desplazamiento por tramo de gap que llevó al siguiente check (cambio del gap). En el informe final: separar el "auto en otra vuelta" según si el piloto **dobla** o **es doblado** (§3c, `final_errors.py`). Así se ve por qué E4 no mejora a E0: corrige el caso del doblado y borra la información del caso en que un líder dobla al piloto, que es donde E0 tiene su mayor error (0,96 s). De ahí sale el siguiente check propuesto, `being_lapped`. También corregir la lectura SHAP del Hito 2 separando la contribución del gap (−50,9 ms) del cambio de predicción (−44,5 ms) (§4a).

### 5c. Uso de IA (C5c)
| Quién | Uso real | Qué se adoptó | Verificación y resultado |
|---|---|---|---|
| Valentín | Claude: leer el enunciado y el feedback en Canvas, proponer el reparto, escribir `hito2_experiment.py`, `final_experiment.py` y el borrador de §1, §2, §5 y §6. Prompt representativo: «ahora viene esto así que divídelo en dos igual y trabaja» | El reparto, la estructura de ambos scripts, los checks K/F, el bootstrap por bloques, el placebo con 200 permutaciones. Las representaciones E3a y E4 salen del feedback del profe y del comentario de la Demo | 23 checks OK en `final_experiment.py`; E0/E1 reproducen el Hito 2 (F5) y la semilla 414 reproduce el placebo del Hito 2 (F10); recalculé el global como promedio ponderado de R8 y R9 (F9) y contrasté cada cifra de §2 con `salidas_final/*.csv` |
| Camilo | Claude (Claude Code): traer los cambios del repo, escribir `final_errors.py`, buscar un caso con el auto de adelante en otra vuelta y redactar el borrador de §3–4. Prompt representativo: «trae los cambios del repo, y termina mi parte que corresponda con lo q se hizo en el repo». La idea de separar doblados viene de mi pregunta tras la Demo: «hay carreras donde los pilotos sacan 1 vuelta a otros y el piloto de delante en realidad va atrás» | El script descriptivo sobre las predicciones de `final_experiment.py` (sin reajustar modelos), la separación doblar / ser doblado, el segundo caso `2026-R8-81-L69` y la tabla que descompone E1 − E0 para la corrección SHAP. Descarté por ahora Ridge/Lasso y los sectores vía API externa: no cambian la pendiente única del OLS y exigen otra fuente | Corrí en mi computador los cuatro scripts en orden: 18 + 13 + 23 + 12 checks OK, y `final_experiment.py` reproduce las salidas de Valentín (diferencias ~1e-15 s). Las restas de los dos casos coinciden con `abs_err_*` (C5); el global de E1 y E4 es igual a R8/R9 ponderado (C1); la dirección doblar/ser doblado reconstruida coincide con `ahead_other_lap` (C3); la suma de la descomposición SHAP da −44,46 ms = 72,6633 − 72,7078 |

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
- **Camilo Bufadel:** hipótesis de tráfico y plan (Hito 1); errores por contexto y SHAP (`hito2_shap.py`, Hito 2); en el informe final, `final_errors.py` y `salidas_final/errors_*` (por carrera E1/E4, doblar vs ser doblado, casos `2026-R8-3-L10` y `2026-R8-81-L69`), §3, §4 (incluida la corrección SHAP −50,9 vs −44,5 ms) y su fila de IA.
