# Hito 2 · NextLap · Experimento ejecutado, errores y SHAP

Equipo: Castro–Bufadel · Integrantes: Valentín Castro, Camilo Bufadel
Repo: https://github.com/Lilvecp/capstone · Commit: ver tag `Hito2` y el texto de entrega · Fecha: 19-11-2026

Comando único: `python hito2_experiment.py` → `salidas_hito2/` (18 checks; ver §4). Todas las cifras de §1 salen de esos archivos.

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

## 2. Errores por contexto  — [Camilo · PENDIENTE]

> Insumos listos: `salidas_hito2/control_vs_candidate_by_race.csv` (n, MAE y diferencia por R8/R9 + global), `validation_predictions.csv` (observado, predicciones y error absoluto por `pair_id`), `mae_by_gap_bin.csv`.
> Falta: tabla R8/R9 (n, MAE control, MAE candidato, control − candidato), explicar si coincide con el global (ponderado por n; K9 ya lo verifica), **un caso real por `pair_id`** con observado, predicción y |error| verificado a mano, y la limitación concreta + siguiente check.

## 3. SHAP  — [Camilo · PENDIENTE]

> Insumos listos: `salidas_hito2/models/E0_reference_ols.joblib` (modelo que retenemos) y `E1_candidate.joblib`; fondo desde `train_features.csv` (solo train); filas de validación en `validation_features.csv`; mapeo de inputs codificados en `candidate_coefficients.csv`.
> Falta: modelo elegido y por qué, una fila de validación (`pair_id`, predicción, base, contribuciones con signo, base + suma = predicción), |error| aparte, y resumen global (media |SHAP| sobre una muestra declarada), explainer y fondo.
> Nota: los `.joblib` dependen de la versión de scikit-learn; regenerarlos con `python hito2_experiment.py` en el mismo entorno donde se corre SHAP.

## 4. Reproducibilidad  — [Valentín]

| Qué | Dónde |
|---|---|
| Datos | `base_NextLap/` = release `IIT414W_Capstone_NextLap_StudentData_v1.zip` sin cambios; SHA-256 en README y verificados por K0a; snapshots verificados contra `source_manifest_v2.json`. Sin test |
| Entorno | `requirements.txt`; versiones exactas de la corrida en `salidas_hito2/run.json` → `environment` |
| Comando | `python hito2_experiment.py` (desde la raíz; ~10 s) |
| Código → salida | ver tabla en `README.md` §Hito 2 |
| Trials | `salidas_hito2/trials_log.csv` (7 corridas: 3 LORO en train + 4 en validación) y `DECISION_LOG.md` |

## 5. Explicaciones individuales

**Valentín Castro.** Decidí usar el OLS de referencia también como modelo de control, porque el candidato era prácticamente el mismo, pero con una variable adicional: el tráfico. Así podía comprobar si esa información realmente mejoraba las predicciones.
Revisé en K2b y K4 que ambos modelos utilizaran las mismas 1.808 filas y que el scaler se ajustara solo con los datos de entrenamiento. La diferencia entre el control y el candidato fue de −0,00034 segundos, con un intervalo de confianza del 95 % entre −0,0027 y +0,0021. Como el intervalo incluye el cero, no podemos asegurar que agregar el gap mejore el modelo, por lo que decidimos descartarlo.
Una limitación es que solo medimos el gap al cruzar la línea en el instante t. Como siguiente paso, probaría utilizar el promedio de las últimas 2 o 3 vueltas, considerando únicamente información disponible en ese momento, para comprobar si el tráfico constante tiene mayor influencia.
Utilicé Claude como apoyo para escribir el script, pero comprobé personalmente que E0 reprodujera el MAE de 0,3867 segundos del Hito 1 (K7). Además, recalculé a mano el MAE global utilizando el promedio ponderado de R8 y R9 (K9), para verificar que los resultados fueran correctos.

**Camilo Bufadel.** *(PENDIENTE: decisión propia, resultado o check que la respalda, una limitación o siguiente check, y uso de IA con su verificación; ~1 minuto de lectura.)*
