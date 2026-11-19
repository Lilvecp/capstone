# Hito 2 · NextLap · Experimento ejecutado, errores y SHAP

Equipo: Castro–Bufadel · Integrantes: Valentín Castro, Camilo Bufadel
Repo: https://github.com/Lilvecp/capstone · Commit: ver tag `Hito2` y el texto de entrega · Fecha: 19-11-2026

Comandos: `python hito2_experiment.py` → `salidas_hito2/` (18 checks; ver §4), y después `python hito2_shap.py` → `salidas_hito2/shap_*` (13 checks). Todas las cifras de §1–3 salen de esos archivos.

## 0. Hipótesis y cambios desde el Hito 1  — [Valentín]

- **Hipótesis (sin cambios):** `gap_ahead_s` = segundos entre que el auto inmediatamente anterior (otro piloto, misma carrera) cruzó la línea y el fin de la vuelta t. Representación pre-declarada: `gap_ahead_cap_s = min(gap_ahead_s, 5 s)`, una numérica más sobre los 6 inputs de la referencia, mismo OLS.
- **Fuente y disponibilidad en t:** `base_NextLap/data/race_2026_r0X_laps_v2.csv` (SHA-256 verificados contra `source_manifest_v2.json`). Solo se usa `end_time_s` de autos con `end_time_s ≤ end_time_s(t)` (check K1: máx. `end_time_ahead − end_time_t` = 0,000 s). Se calcula sobre la **línea de tiempo original completa** (todas las vueltas, incluidas pit/SC/no precisas) **antes** del filtro `clean_pair`, y se une a los CSV publicados por `pair_id` (K2a: 5.772 → 5.772 y 1.808 → 1.808, sin duplicados). Si se filtrara primero, 503 pares clean tendrían otro gap (K3b).
- **Faltantes:** regla pre-declarada = tope 5 s ("aire libre"). Cobertura 100 %: **0 filas imputadas** en train y en validación (K5, `feature_coverage.csv`). La feature no usa historial del stint, así que no hay regla de historial corto.
- **Cambio respecto del Hito 1 (corrección por feedback C1.3):** en HITO1 §1 escribimos que `clean_pair` deja fuera el tráfico fuerte. Eso es incorrecto: `clean_pair` filtra pit, banderas, vueltas no precisas/borradas y cambio de stint, pero **no** tráfico. Evidencia: el 18,1 % de los pares de validación (328/1.808) tiene un auto a < 1 s. La frase queda corregida aquí y en el informe final.

## 1. Comparación ejecutada y ablación  — [Valentín]

Mismas 1.808 filas para todas las versiones (hash `f453a98d…`, K2b), split R1–7 / R8–9 (K0b), preprocesamiento ajustado solo con train (K4), métrica MAE en segundos.

| Versión | Modelo e inputs | n val | MAE (s) | Salida |
|---|---|---|---|---|
| OLS referencia **= control pareado** (misma corrida) | OLS `make_model`, 6 inputs publicados | 1.808 | **0,38671** | `validation_predictions.csv` · `pred_E0_reference_ols` |
| Candidato | mismo OLS, 6 inputs + `gap_ahead_cap_s` | 1.808 | **0,38705** | `pred_E1_candidate` |
| Placebo (control extra) | mismo OLS, 6 + `gap_ahead_cap_s` permutado dentro de cada carrera (seed 414) | 1.808 | 0,38678 | `pred_E2_placebo` |
| Sensibilidad S1 | mismo OLS, 6 + gap excluyendo autos en vuelta de pit in/out | 1.808 | 0,38731 | `pred_S1_sensitivity_nopit` |

Como el candidato es OLS con los 6 inputs + la feature, el **OLS de referencia es el control pareado**: misma familia, mismos hiperparámetros (ninguno), mismo `Pipeline`, mismas filas y misma métrica; solo se quita `gap_ahead_cap_s` y se reajusta. El E0 reproduce el OLS del Hito 1 (0,3867 s, K7). Fuente: `comparison_table.csv`.

**Control − candidato = 0,38671 − 0,38705 = −0,00034 s.** El signo negativo significa que el candidato tiene un MAE **mayor** (peor) que el control en estas filas. IC 95 % bootstrap por bloques piloto-carrera (43 bloques, 2.000 remuestreos, seed 414): **[−0,0027 ; +0,0021] s**, incluye 0 (`run.json`). El candidato tiene menor error absoluto en solo el 48,8 % de las filas.

**Qué apoya sobre la hipótesis.** Con los criterios fijados antes de implementar (HITO1 §4), el resultado **contradice** la hipótesis: E1 ≥ E0. Añadir la distancia al auto de adelante, en esta representación lineal con tope, no reduce el MAE de la referencia. El placebo (−0,00008 s) y S1 (−0,0006 s) tampoco mejoran: la columna nueva no aporta más que ruido, y quitar los autos en pit no cambia la conclusión.
**Límite de la afirmación.** El resultado vale para OLS, la representación `min(gap, 5 s)`, la población `clean_pair` y dos carreras (R8–R9). No descarta que el tráfico importe en general: el gap se mide en la línea al final de t (no dice si el auto sigue cerca en t+1), y una relación no lineal o una interacción con el circuito no se probaron.

**Decisión: se descarta la feature** y se mantiene la referencia OLS de 6 inputs como modelo del equipo (DECISION_LOG).

### Checks definidos antes de implementar (HITO1 §4) y resultado observado

| Check (Hito 1, sin editar) | Resultado Hito 2 | Archivo |
|---|---|---|
| K1 Temporal: `end_time_s` del auto usado ≤ `end_time_s(t)` | OK, máx. diferencia 0,000 s; nunca el mismo piloto (K1b) | `checks.csv` |
| K2 Mismas filas: hash `f453a98d…`, n = 1.808, mismo orden | OK (K2a, K2b, K8) | `checks.csv` |
| K3 Sin fuga: sin `next_*`; "falla si la feature cambia al filtrar" | OK, sin `next_*` ni `clean_pair` como input (K3a). **Revisado**, ver abajo (K3b, K3c) | `checks.csv` |
| K4 Media del scaler de la feature = media de train | OK: 2,7812 (train) vs 2,7875 (train+val) | `checks.csv` |
| K5 Contar filas imputadas con el tope | 0 en train, 0 en validación | `feature_coverage.csv` |
| K6 Placebo dentro de carrera, seed 414, misma distribución | OK | `checks.csv` |
| Criterio "apoya": E1 < E0 global y en R8 y R9; E2 sin mejora; mejora concentrada en gap < 2 s | **No se cumple**: E1 > E0 global; R8 −0,0019 s, R9 +0,0019 s; mejora solo en gap ≥ 2 s (ver tabla) | `control_vs_candidate_by_race.csv`, `mae_by_gap_bin.csv` |

*Revisión del procedimiento (se conserva la entrada original arriba).* K3 decía "falla si la feature cambia al filtrar". Al implementarlo vimos que la feature **debe** cambiar si se filtra primero (503 pares), precisamente porque se calcula sobre la línea completa: el check correcto es demostrar que se construye **antes** del filtro (K3b) y que el fin de t usado coincide con el publicado (K3c). Agregamos K0 (datos/split), K7 (reproduce Hito 1), K8 (predicciones finitas) y K9 (MAE global = R8/R9 ponderado por n).

Tramo de gap (slice pre-declarado, mismas filas):

| Gap al auto de adelante | n | MAE control (s) | MAE candidato (s) | Control − candidato (s) |
|---|---|---|---|---|
| < 1 s | 328 | 0,4516 | 0,4574 | −0,0059 |
| 1–2 s | 439 | 0,3978 | 0,4010 | −0,0033 |
| 2–5 s | 590 | 0,3621 | 0,3611 | +0,0009 |
| ≥ 5 s | 451 | 0,3610 | 0,3561 | +0,0049 |

El patrón es el contrario al esperado: donde hay tráfico (< 2 s) el candidato empeora. Ese era el patrón que declaramos "sospechoso" en el Hito 1.

### Selección (antes de validación, solo train)

Leave-one-race-out dentro de R1–7 (`selection_loro_train.csv`): E0 0,50839 · E1 (tope 5 s) 0,50858 · E1b (indicador < 1 s) 0,50799 s. E1b no supera a E1 por más de 0,005 s, así que por la regla pre-declarada se eligió **E1**. Ya en train la feature no mejoraba a E0 (señal temprana, registrada en `trials_log.csv`). Validación R8–9 se usó **una sola vez** (4 corridas).

## 2. Errores por contexto  — [Camilo]

Comando: `python hito2_shap.py` (después de `hito2_experiment.py`) → `salidas_hito2/shap_case_error.csv`, `shap_context_by_gap_bin.csv`. Tabla por carrera desde `control_vs_candidate_by_race.csv`.

| Carrera | n | MAE control E0 (s) | MAE candidato E1 (s) | Control − candidato (s) | Filas donde E1 gana |
|---|---|---|---|---|---|
| R8 Austrian GP | 1.066 | 0,37357 | 0,37549 | **−0,0019** | 47,5 % |
| R9 British GP | 742 | 0,40558 | 0,40365 | **+0,0019** | 50,8 % |
| Global R8–R9 | 1.808 | 0,38671 | 0,38705 | −0,00034 | 48,8 % |

**¿Coincide con el global?** No del todo, y eso es lo informativo. En R8 gana el control y en R9 el candidato, con diferencias casi iguales y de signo opuesto. El global es el promedio ponderado por n: (1.066 · −0,00192 + 742 · +0,00193) / 1.808 = −0,00034 s (lo verifica K9). Queda casi en 0 porque R8 tiene más filas, no porque las dos carreras se parezcan. Ninguna diferencia por carrera llega a 0,01 s, frente a un MAE de ~0,39 s, y el IC global incluye 0. **No es una mejora consistente**: el candidato no gana en las dos carreras, que era uno de los criterios "apoya" del Hito 1.

**Caso real: `2026-R8-3-L10`** (Austria, piloto 3, Red Bull Racing, vuelta 10 → 11; MEDIUM con 10 vueltas). Se eligió porque tiene el auto de adelante a **0,644 s** (tramo < 1 s, donde el candidato empeora más), un error grande y el candidato peor que el control.

| | Valor (s) |
|---|---|
| Vuelta t (`current_lap_s`) | 72,875 |
| Observado t+1 (`y_true_s`) | **74,413** (1,54 s más lenta) |
| Predicción control E0 | 72,708 → \|74,413 − 72,708\| = **1,705** |
| Predicción candidato E1 | 72,663 → \|74,413 − 72,663\| = **1,750** |
| Control − candidato | 1,705 − 1,750 = **−0,044** (empeora) |

Las dos restas coinciden con `abs_err_*` de `validation_predictions.csv` (check S2). Ninguno de los dos modelos anticipa una vuelta 1,5 s más lenta. El candidato, además, mueve la predicción 0,044 s hacia "más rápido" porque el gap es chico, es decir, en la dirección contraria a lo que pasó.

**El caso no es una excepción** (`shap_context_by_gap_bin.csv`):

| Tramo de gap | n | Desplazamiento medio E1 − E0 (s) | % filas con t+1 más lenta que E0 | Control − candidato si t+1 más lenta | … si t+1 más rápida |
|---|---|---|---|---|---|
| < 1 s | 328 | −0,046 | 54,3 % | −0,046 | +0,042 |
| 1–2 s | 439 | −0,033 | 55,4 % | −0,032 | +0,032 |
| 2–5 s | 590 | +0,010 | 55,4 % | +0,009 | −0,009 |
| ≥ 5 s | 451 | +0,052 | 56,5 % | +0,048 | −0,051 |

Con gap chico, el coeficiente positivo de `gap_ahead_cap_s` (`candidate_coefficients.csv`) baja la predicción (en promedio 0,046 s en el tramo < 1 s). Eso solo ayuda cuando la vuelta siguiente es efectivamente más rápida. En < 1 s, la vuelta t+1 fue más lenta que la predicción de E0 en el 54 % de las filas, así que el desplazamiento se equivoca de dirección más veces de las que acierta y la ganancia y la pérdida se compensan.

**Limitación concreta.** Un gap de 0,6 s en la línea **no distingue** entre "voy a adelantar o tengo DRS" (vuelta más rápida) y "quedo atascado detrás" (vuelta más lenta, como en `2026-R8-3-L10`). El OLS con una sola pendiente lineal aplica el mismo desplazamiento en los dos casos. Esto vale para R8–R9, `clean_pair` y la representación `min(gap, 5 s)`.
**Siguiente check (mismo protocolo, mismas 1.808 filas, selección en train):** separar esas dos situaciones con información disponible en t, mediante (a) el **cambio del gap** entre t−1 y t (¿se está acercando o está estancado detrás?) y (b) el gap medio de las últimas 2–3 vueltas (tráfico persistente, la propuesta de §5 de Valentín). Si (a) aporta, debería verse primero en el tramo < 1 s y en las filas donde t+1 fue más lenta.

## 3. SHAP  — [Camilo]

Comando: `python hito2_shap.py` → `salidas_hito2/shap_local.csv`, `shap_global.csv`, `shap_checks.csv` (13 checks), `shap_run.json` (explainer, fondo, muestra y versiones: Python 3.12.13, sklearn 1.9.1, shap 0.54.1).

**Modelo explicado: E0 (referencia OLS)**, porque es el que retenemos según la ablación (§1). Para mostrar cuánto pesa el gap también corrí E1 y lo reporto aparte, sin usarlo como modelo del equipo.
**Explainer y fondo.** `shap.LinearExplainer` (masker `Independent`) sobre la regresión ya preprocesada. Fondo: **1.000 filas de `train_features.csv` (solo R1–7), `sample(random_state=414)`**, transformadas con `model.named_steps['prepare']` (ajustado solo con train). Las columnas `numeric__*` y las one-hot `category__*` se suman de vuelta a su input original (`current_compound`, `current_team`); en un modelo lineal la suma conserva la aditividad. Contribuciones en segundos de `next_lap_s`. Los `.joblib` se regeneraron en mi entorno y reproducen `pred_E0`/`pred_E1` con diferencia < 1e-6 s (S1).

**Local: `2026-R8-3-L10`** (el mismo caso de §2), E0:

| Input | Valor en la fila | Contribución SHAP (s) |
|---|---|---|
| base value (predicción media del fondo) | — | **87,2052** |
| `recent_mean_s` | 72,610 | −8,0462 |
| `current_lap_s` | 72,875 | −6,4044 |
| `current_tyre_life` | 10 | −0,0259 |
| `current_team` | Red Bull Racing | −0,0145 |
| `current_compound` | MEDIUM | −0,0055 |
| `current_lap_number` | 10 | −0,0009 |
| **base + suma** | | **72,7078** |
| predicción del modelo | | 72,7078 ✓ (diferencia < 1e-6, check S6) |
| *aparte:* observado / \|error\| | | 74,413 / 1,705 |

*Lectura del signo (asociación del modelo, no causalidad).* Las contribuciones se leen respecto de la base de 87,2 s, que es la vuelta media que predice el modelo en el fondo de R1–7, una mezcla de circuitos más largos. Casi todo el descenso hasta 72,7 s viene de los dos inputs de ritmo (−14,45 s entre ambos): el modelo "reconoce" que es una vuelta corta de Austria, no algo propio de este piloto. Neumático de 10 vueltas, Red Bull y MEDIUM restan cada uno menos de 0,03 s frente al promedio del fondo. Ninguna contribución empuja hacia una vuelta más lenta, así que el modelo no tenía con qué anticipar los +1,54 s observados. En **E1**, para la misma fila, `gap_ahead_cap_s` (0,644 s) contribuye **−0,051 s**: es lo que baja la predicción a 72,663 y sube el \|error\| a 1,750 s. El resto de contribuciones cambia algo porque E1 se reajustó.

**Global: media \|SHAP\| por input sobre las 1.808 filas de validación R8–9** (todas, sin muestreo):

| Rank | Input (E0) | Media \|SHAP\| (s) | % del total | | Input (E1) | Media \|SHAP\| (s) |
|---|---|---|---|---|---|---|
| 1 | `recent_mean_s` | 6,509 | 54,7 % | | `recent_mean_s` | 6,424 |
| 2 | `current_lap_s` | 5,273 | 44,3 % | | `current_lap_s` | 5,364 |
| 3 | `current_compound` | 0,048 | 0,4 % | | `current_compound` | 0,048 |
| 4 | `current_team` | 0,033 | 0,3 % | | **`gap_ahead_cap_s`** | **0,037** |
| 5 | `current_tyre_life` | 0,028 | 0,2 % | | `current_team` | 0,033 |
| 6 | `current_lap_number` | 0,001 | 0,0 % | | `current_tyre_life` | 0,026 |
| 7 | | | | | `current_lap_number` | 0,007 |

El modelo se apoya casi por completo (99 %) en el ritmo de la vuelta actual y en la media reciente. En E1, el gap mueve la predicción en promedio 0,037 s, un orden de magnitud menos que el MAE (0,387 s). Es del mismo tamaño que los desplazamientos de §2 y, como allí, no se traduce en menor error.

**Límites.** (1) Solo esta muestra: el fondo es R1–7 y las filas explicadas son R8–R9 `clean_pair`. Como la base mezcla circuitos, buena parte de la \|SHAP\| de los inputs de ritmo es "nivel del circuito frente al fondo", no dinámica vuelta a vuelta; otro fondo (por ejemplo, el mismo circuito) daría magnitudes distintas. (2) SHAP describe en qué se apoya el **modelo**, no qué causa el tiempo de vuelta. (3) Una contribución grande no prueba menor error: el gap tiene \|SHAP\| > 0 en E1 y aun así E1 no mejora el MAE (§1–2). (4) El masker `Independent` trata los inputs como independientes, y `current_lap_s` y `recent_mean_s` están muy correlacionados, así que el reparto entre ellos es una convención y no algo identificado.

## 4. Reproducibilidad  — [Valentín]

| Qué | Dónde |
|---|---|
| Datos | `base_NextLap/` = release `IIT414W_Capstone_NextLap_StudentData_v1.zip` sin cambios; SHA-256 en README y verificados por K0a; snapshots verificados contra `source_manifest_v2.json`. Sin test |
| Entorno | `requirements.txt`; versiones exactas de la corrida en `salidas_hito2/run.json` → `environment` |
| Comando | `python hito2_experiment.py` (desde la raíz; ~10 s) y después `python hito2_shap.py` (§2–3; usa los `.joblib` recién generados) |
| Código → salida | ver tabla en `README.md` §Hito 2 |
| Trials | `salidas_hito2/trials_log.csv` (7 corridas: 3 LORO en train + 4 en validación) y `DECISION_LOG.md` |

## 5. Explicaciones individuales

**Valentín Castro.** Decidí usar el OLS de referencia también como modelo de control, porque el candidato era prácticamente el mismo, pero con una variable adicional: el tráfico. Así podía comprobar si esa información realmente mejoraba las predicciones.
Revisé en K2b y K4 que ambos modelos utilizaran las mismas 1.808 filas y que el scaler se ajustara solo con los datos de entrenamiento. La diferencia entre el control y el candidato fue de −0,00034 segundos, con un intervalo de confianza del 95 % entre −0,0027 y +0,0021. Como el intervalo incluye el cero, no podemos asegurar que agregar el gap mejore el modelo, por lo que decidimos descartarlo.
Una limitación es que solo medimos el gap al cruzar la línea en el instante t. Como siguiente paso, probaría utilizar el promedio de las últimas 2 o 3 vueltas, considerando únicamente información disponible en ese momento, para comprobar si el tráfico constante tiene mayor influencia.
Utilicé Claude como apoyo para escribir el script, pero comprobé personalmente que E0 reprodujera el MAE de 0,3867 segundos del Hito 1 (K7). Además, recalculé a mano el MAE global utilizando el promedio ponderado de R8 y R9 (K9), para verificar que los resultados fueran correctos.

**Camilo Bufadel.** Expliqué con SHAP el modelo E0 y no el candidato. Era la sugerencia del reparto y la acepté porque es el que el equipo retiene según la ablación; el candidato lo usé solo para ver cuánto pesaba el gap. Para el análisis de errores usé el par `2026-R8-3-L10`, sugerido por Claude, con el auto de adelante a 0,644 s y una vuelta siguiente 1,54 s más lenta, donde el candidato empeora el error de 1,705 a 1,750 s.
Lo que respalda la lectura es que el caso no es aislado: con gap < 1 s, el candidato baja la predicción en promedio 0,046 s, pero en el 54 % de esas 328 filas la vuelta siguiente fue más lenta, así que ese ajuste se equivoca de dirección más de lo que acierta. En R8 gana el control (−0,0019 s) y en R9 el candidato (+0,0019 s), por lo que no hay una mejora consistente. En SHAP, el gap aporta en promedio 0,037 s, contra 0,387 s de MAE.
Una limitación es que el gap en la línea no distingue entre "voy a adelantar" y "quedo atascado detrás". Como siguiente paso, probaría el cambio del gap entre t−1 y t, con el mismo protocolo.
Usé Claude Code para escribir `hito2_shap.py` y redactar §2 y §3. Lo verifiqué así: comprobé las dos restas del caso (74,413 − 72,708 = 1,705 y 74,413 − 72,663 = 1,750 s); el script comprueba que base + suma = predicción en las 1.808 filas y que cada SHAP es igual a coef · (x − media del fondo), calculado aparte (13 checks); y revisé cada cifra del texto contra los CSV de `salidas_hito2/`.
