# DECISION_LOG · Hito 1 NextLap

| Fecha | Quién | Decisión | Evidencia | Cambio posterior / siguiente paso |
|---|---|---|---|---|
| 2026-11-11 | Equipo | Reparto del trabajo en dos mitades: Valentín §1–2 y empaquetado; Camilo §3–4 | Rúbrica Hito 1 v2 | |
| 2026-11-11 | Valentín | Script propio `hito1_reference.py` que reutiliza el helper del curso en vez de entregar solo `probar_referencia_v1.py` | La rúbrica pide reproducción propia; la salida del instructor no cuenta | Se agregaron 9 checks (SHA-256, conteos, rondas, `pair_id` disjuntos, orden temporal, solo clean, sin `next_*`, scaler ajustado solo en train, mismas filas) |
| 2026-11-11 | Valentín | Reportar MAE por carrera y comparación pareada OLS vs persistencia | `salidas_hito1/paired_ols_vs_persistence.csv`: OLS −0,023 s, pero gana solo en el 51,6 % de las filas | La mejora de la referencia es modesta: cualquier extensión del Hito 2 debe compararse contra OLS en las mismas filas y por carrera |
| 2026-11-11 | Valentín | Guardar el hash del set de `pair_id` de validación (`f453a98d…`) | `reference_run.json` | En el Hito 2, verificar que la extensión y la ablación usen exactamente ese set |
| 2026-11-11 | Valentín | Cuantificar el límite de `clean_pair` con `observed_pairs_train_validation_v1.csv` | 82,4 % (R8) y 71,8 % (R9) de los pares observados son clean | Reportar por separado cualquier análisis sobre la población amplia |
| 2026-11-12 | Camilo | Reproducir la referencia en una segunda máquina antes de planificar | `python hito1_reference.py` con Python 3.12.13 / pandas 3.0.6 / numpy 2.5.3 / sklearn 1.9.1: 9 checks OK y los mismos MAE (0,3867 / 0,4093 / 11,3761 s); solo cambian dígitos de punto flotante (~1e-14) | No se subieron esas salidas para conservar las de la entrega original |
| 2026-11-12 | Camilo | Elegir **tráfico** (`gap_ahead_s` desde `end_time_s`) como hipótesis principal, en vez de variabilidad de ritmo (el ejemplo resuelto en clase) o clima (requiere fuente externa) | Los snapshots tienen las vueltas de todos los pilotos con el mismo reloj: el gap se puede calcular sin fuente externa y con una regla temporal que se puede revisar | Si no aporta, se reporta el resultado negativo; la alternativa pre-declarada es el indicador `gap < 1 s` |
| 2026-11-12 | Camilo | Calcular la feature sobre la línea de tiempo completa y **antes** del filtro `clean_pair` | El brief pide construir la historia antes del filtrado retrospectivo; filtrar primero borraría autos que sí estaban en pista (pit/SC) | Check F4–F6: los `pair_id` reconstruidos coinciden con el release |
| 2026-11-12 | Camilo | Script de factibilidad sin modelos ni target de validación | `salidas_hito1/feasibility_traffic_*`: 7 checks OK, 0 faltantes, 22,9 % de train con gap < 1 s | Escribir los checks K1–K6 antes de implementar E1 (HITO1 §4) |
| 2026-11-12 | Camilo | Fijar el tope en 5 s y como máximo 2 variantes, elegidas con LORO dentro de train; validación una sola vez | Una búsqueda grande sobre validación no es evidencia fuerte (brief) | Registrar todos los intentos en `trials_log.csv` (Hito 2) |
| 2026-11-12 | Camilo | Agregar el placebo (gap permutado dentro de la carrera) además de la ablación E1 vs E0 | Separa "información de tráfico" de "una columna más" | — |

## Revisión tras feedback

- Entrada original (HITO1 §1): el momento de predicción decía solo "al final de la vuelta t", sin definir qué información cuenta como disponible en ese instante.
- Feedback recibido: no quedaba claro **cuándo** se predice ni qué datos se pueden usar en ese momento.
- Entrada revisada: "Al terminar la vuelta t (`end_time_s` de t), usando solo información observada hasta ese instante y dentro del mismo piloto/carrera/stint" (HITO1 §1, fila "Momento de predicción"). Esta misma regla es la que después se usa para la hipótesis: `gap_ahead_s` solo usa autos con `end_time_s ≤ end_time_s(t)` (check F2).
- Se mantuvo sin cambios: la referencia, la hipótesis y el diseño del experimento. El feedback no pedía cambiarlos y ya cubrían disponibilidad en t, mismas filas de evaluación, cambio aislado y checks escritos antes de implementar.

## Uso de IA

| Quién | Herramienta | Para qué | Sugerencia adoptada | Cómo se verificó |
|---|---|---|---|---|
| Valentín | Claude | Apoyo para ordenar el enunciado y los materiales, y como punto de partida para el reparto y las plantillas | La idea de repartir por criterio de rúbrica y una estructura base para HITO1/DECISION_LOG. El reparto concreto se definió en equipo (fila 2026-11-11) | Contrastado con el brief y la rúbrica v2 de Canvas |
| Valentín | Claude | Apoyo en la escritura de `hito1_reference.py` y del README, y borrador de redacción de HITO1 §1–2 | Estructura del script y redacción base. Qué checks incluir, la comparación pareada, el hash de validación y la cuantificación de `clean_pair` los definí yo (filas 2026-11-11), igual que la interpretación de los resultados | Ejecuté el script en mi computador (Python 3.12, numpy 2.5.4, pandas 3.1.1, sklearn 1.9.1): 9 checks OK y MAE iguales a `salidas_referencia/` del instructor (0,3867 / 0,4093 / 11,3761 s). Revisé cada cifra del texto contra `salidas_hito1/` y los porcentajes de `clean_pair` contra `observed_pairs_train_validation_v1.csv` |
| Camilo | Claude (Claude Code) | Revisar el repo y el brief, verificar la factibilidad de la hipótesis de tráfico, redactar `hito1_feasibility_traffic.py` y el borrador de HITO1 §3–5 | Elegir tráfico desde `end_time_s`, el tope de 5 s, el placebo por permutación y la selección LORO en train | Ejecución de los dos scripts en mi computador: 9 + 7 checks OK. Las cifras de §3 salen de `salidas_hito1/feasibility_traffic_coverage.csv` y `_train_by_round.csv`. Contrastamos los comentarios de la IA con el brief y la rúbrica v2, punto por punto. |
| Equipo | Claude (Claude Code) | Revisar la coherencia y la estructura de la entrega completa (HITO1, DECISION_LOG, README) | Nos basamos en sus comentarios para cumplir los requisitos que nos propusimos: secciones del brief completas, cifras consistentes entre documentos y salidas, y los placeholders pendientes | Cada comentario se contrastó con el brief Hito 1 v2 y la rúbrica; las cifras citadas se revisaron contra `salidas_hito1/` |

## Contribuciones reales

- Valentín: encuadre del problema y límite de `clean_pair` (HITO1 §1); reproducción de las tres referencias con checks y la interpretación (HITO1 §2, `hito1_reference.py`, `salidas_hito1/`); README, entorno y empaquetado final.
- Camilo Bufadel: segunda reproducción de la referencia; hipótesis de tráfico, su fuente y regla temporal (HITO1 §3); plan de experimento, ablación, placebo y checks K1–K6 (HITO1 §4); `hito1_feasibility_traffic.py` y sus salidas.

## Antes del Hito 2

- Camilo: implementar E1/E2/S1 con los checks K1–K6, la selección LORO en train y `trials_log.csv`.
- Valentín: MAE por carrera y por tramo de gap con conteos, IC bootstrap de la diferencia pareada, población amplia por separado y SHAP.
- Bloqueo conocido: ninguno técnico. Pendiente consultar con Ángel cómo tratar los autos en pit lane (S1).

---

# DECISION_LOG · Hito 2

| Fecha | Quién | Decisión | Evidencia | Cambio posterior / siguiente paso |
|---|---|---|---|---|
| 2026-11-18 | Equipo | Reparto Hito 2: Valentín = pipeline del experimento (feature, referencia, candidato, control pareado, placebo, S1, checks, reproducibilidad); Camilo = errores por contexto (§2) y SHAP (§3) sobre esas salidas | Rúbrica Hito 2 v2 (C1–C3, C7 / C4–C6) | Cambia el plan del Hito 1 (allí Camilo implementaba E1/E2/S1): se invirtió para que el análisis de errores y SHAP lo haga quien no escribió el pipeline |
| 2026-11-18 | Valentín | Usar el OLS de referencia como control pareado (candidato = mismo OLS + 1 columna) | Brief §1: "the OLS reference also serves as the matched control" | Una sola corrida E0 cumple ambos roles (`comparison_table.csv`) |
| 2026-11-18 | Valentín | Selección E1 vs E1b con LORO en train antes de mirar validación | `selection_loro_train.csv`: E0 0,50839 · E1 0,50858 · E1b 0,50799 s → E1 por la regla de 0,005 s | Ya en train E1 no mejoraba a E0 (trial no exitoso, registrado) |
| 2026-11-18 | Valentín | Validación una sola vez: E0, E1, E2 placebo, S1 | `comparison_table.csv`: 0,38671 / 0,38705 / 0,38678 / 0,38731 s | — |
| 2026-11-18 | Valentín | **Descartar `gap_ahead_cap_s`** y retener la referencia de 6 inputs | Control − candidato = −0,00034 s, IC 95 % [−0,0027; +0,0021] (`run.json`); R8 −0,0019, R9 +0,0019; criterios "apoya" del Hito 1 no se cumplen | Siguiente check: gap promedio de las últimas 2–3 vueltas (tráfico persistente, disponible en t) en el mismo protocolo |
| 2026-11-18 | Valentín | Revisar el check K3 ("falla si la feature cambia al filtrar") | Al implementarlo, 503 pares clean cambian de gap si se filtra primero: eso confirma que hay que calcular antes del filtro, no es una falla | Se conserva K3 original en HITO2 §1 y se reemplaza por K3b/K3c; se agregan K0, K7, K8, K9 |
| 2026-11-18 | Camilo | Explicar con SHAP el modelo **E0** (retenido) y reportar E1 aparte solo para ver el peso del gap | La ablación descarta la feature (§1); explicar el modelo descartado como modelo del equipo sería incoherente | `shap_global.csv`: gap en E1 = 0,037 s de media \|SHAP\| frente a 0,387 s de MAE |
| 2026-11-18 | Camilo | Fondo SHAP = 1.000 filas de train (seed 414) transformadas con el `prepare` ajustado en train; global sobre las 1.808 filas de validación | Brief: fondo sin validación; muestra declarada | Checks S3–S6: aditividad < 1e-6 en todas las filas y SHAP = coef·(x − media fondo) recalculado aparte en el script (S4) |
| 2026-11-18 | Camilo | Caso de §2 = `2026-R8-3-L10` (gap 0,644 s, \|error\| E1 1,750 vs E0 1,705 s) y medir hacia dónde mueve E1 la predicción por tramo | `shap_context_by_gap_bin.csv`: en < 1 s E1 baja la predicción 0,046 s, pero en el 54,3 % la vuelta t+1 fue más lenta que lo predicho por E0 | Siguiente check: cambio del gap t−1→t (acercándose vs. estancado), con el mismo protocolo y selección en train |
| 2026-11-18 | Valentín | S1 implementado con el auto de adelante que no está en vuelta de pit in/out | MAE S1 0,38731 s: no cambia la conclusión | Duda de Hito 1 sobre autos en pit lane queda resuelta para este resultado |

## Revisión tras feedback (Hito 1 → Hito 2)

- Feedback C1.3 (instructor, Hito 1): "clean_pair does not independently exclude strong traffic".
- Entrada original (HITO1 §1): "Las vueltas con safety car, pit o tráfico fuerte quedan fuera".
- Verificación: el 18,1 % de los pares de validación (328/1.808) y el 22,9 % de train tienen un auto a < 1 s (`salidas_hito2/feature_coverage.csv`, `mae_by_gap_bin.csv`).
- Cambio: la afirmación se corrige en HITO2 §0 (y en el informe final): `clean_pair` excluye pit, banderas, vueltas no precisas/borradas y cambio de stint, **no** tráfico. El análisis por tramo de gap se mantiene (HITO2 §1).

## Uso de IA · Hito 2

| Quién | Herramienta | Para qué | Qué se adoptó / cambió | Cómo se verificó |
|---|---|---|---|---|
| Valentín | Claude | Utilicé Claude como apoyo para organizar mis ideas, revisar los requisitos del Hito 2 y facilitar la implementación del código y la redacción. Aprobé el reparto propuesto con Camilo y definí qué experimentos realizar y cómo presentar los resultados. Prompt representativo: «sepáralo para dos personas yo y Camilo, y haz todo lo mío».
Decidí mantener el OLS de referencia como control y evaluar el aporte de la variable de tráfico. Con apoyo de Claude implementé el script hito2_experiment.py, los checks K0–K9 y el bootstrap por bloques. Mantuve los criterios establecidos en el Hito 1, incluyendo la feature, el tope de 5 s y la regla de selección.Ejecuté el script en mi computador y comprobé que E0 reprodujera el MAE del Hito 1 (0,386707 s, K7). También verifiqué el MAE global mediante el promedio ponderado de R8 y R9 (K9), y contrasté las cifras de HITO2 §1 con los archivos CSV generados.|
| Camilo | Claude (Claude Code) | Escribir `hito2_shap.py` (caso de §2, contexto por tramo, SHAP local/global) y el borrador de HITO2 §2–3. Prompt representativo: «revisa y busca si se pueden mejorar algunas cosas» | Adopté sus sugerencias: explicar E0 (ya propuesto en `REPARTO_HITO2.md`; lo acepté porque es el modelo retenido por la ablación), el caso `2026-R8-3-L10` (gap < 1 s, error grande y candidato peor que control), el explainer lineal con fondo de train, la agregación de one-hot por input y el análisis de dirección del desplazamiento E1 − E0 | Corrí `hito2_experiment.py` y `hito2_shap.py` en mi computador (Python 3.12.13, sklearn 1.9.1, shap 0.54.1): 18 + 13 checks OK; los `.joblib` reproducen `pred_E0/E1` (< 1e-6). Comprobé las dos restas del caso (74,413 − 72,708 = 1,705 y 74,413 − 72,663 = 1,750 s; el script las contrasta con `abs_err_*`, S2) y el global ponderado de R8/R9 (−0,00034 s, K9). Revisé cada cifra de §2–3 contra `salidas_hito2/` |

## Contribuciones reales · Hito 2

- Valentín: `hito2_experiment.py` (feature, selección LORO, E0/E1/E2/S1, checks K0–K9, predicciones por `pair_id`, bootstrap, tramos de gap); HITO2 §0, §1, §4; README Hito 2; corrección C1.3. Puede explicar: por qué E0 es el control pareado, por qué la diferencia negativa descarta la feature y por qué K3 se revisó.
- Camilo: `hito2_shap.py` y `salidas_hito2/shap_*`; HITO2 §2 (tabla R8/R9, caso `2026-R8-3-L10`, desplazamiento por tramo, limitación y siguiente check) y §3 (SHAP local y global de E0, gap en E1 aparte, límites); su párrafo en §5. Puede explicar: por qué el global queda en ~0 sin que R8 y R9 coincidan, por qué el gap empeora en tráfico y qué significa (y qué no) la base de 87,2 s.

---

# DECISION_LOG · Informe final

## Pre-registro del check pedido en el feedback del Hito 2 (escrito ANTES de evaluar; 2026-11-22)

Feedback del instructor (Hito 2, comprobación para el informe final): *"Fijen antes de evaluar una representación del cambio del gap entre t−1 y t. Compárenla con E0 sobre los mismos IDs, con selección dentro de R1–R7. Comprueben si reduce el error en el tramo < 1 s y reporten el MAE total y por carrera."*

| Elemento | Definición fijada antes de mirar resultados |
|---|---|
| Feature | `gap_delta_s(t) = gap_ahead_cap_s(t) − gap_ahead_cap_s(t−1)`. t−1 = vuelta anterior **del mismo piloto, misma carrera y mismo stint** (`lap_number = t − 1`). Ambos gaps con tope 5 s, calculados como en el Hito 2 sobre la línea de tiempo completa **antes** del filtro `clean_pair`. Negativo = el auto de adelante está más cerca que al final de t−1. Usa solo cruces con `end_time_s ≤ end_time_s(t)` |
| Faltantes / historial corto | Si no existe la vuelta t−1 en el mismo stint (primera vuelta del stint o vuelta faltante): `gap_delta_s = 0` (sin evidencia de cambio). Regla fija; se reporta cuántas filas la usan en train y validación |
| Variantes (máx. 2) | **E3a** = 6 inputs + `gap_delta_s`. **E3b** = 6 inputs + `gap_ahead_cap_s` + `gap_delta_s` |
| Selección | Leave-one-race-out **dentro de R1–R7**. Se elige E3b solo si su MAE LORO medio es menor que el de E3a por más de 0,005 s; si no, E3a (más simple). Validación R8–R9 se usa una vez para la variante elegida |
| Control pareado | E3a → **E0** (OLS 6 inputs). E3b → **E1** (OLS 6 + `gap_ahead_cap_s`), y además se reporta contra E0. Mismo `make_model('OLS')`, mismas 1.808 filas (hash `f453a98d…`) |
| Reporte | MAE total y por carrera (R8, R9) con n, control − candidato, IC 95 % bootstrap por bloques piloto-carrera (seed 414), y MAE por tramo de gap con foco en < 1 s |
| Apoya | MAE(candidato) < MAE(control) global **y** en R8 y R9, **y** control − candidato > 0 en el tramo < 1 s |
| Contradice | Cualquier otro resultado. Se reporta igual, sin cambiar la feature ni la regla después de mirar validación |
| Placebo (corrección prioritaria) | Además de la permutación única del Hito 2, se repite el placebo de E1 con **200 permutaciones** dentro de carrera (semillas 414–613) para ubicar la permutación observada dentro de esa distribución. La lectura del Hito 2 se acota a la permutación observada |

**Nota de transparencia (2026-11-22).** El pre-registro de arriba se escribió en este archivo a las 14:14 (hora de Santiago, mtime del archivo) y `final_experiment.py` se ejecutó sobre validación a las 14:16, **antes** de que el pre-registro quedara en un commit. Ningún elemento del pre-registro (feature, regla de faltantes, variantes, regla de selección, criterios) se cambió después de ver resultados; el commit que lo incluye lo muestra sin diferencias.

## Pre-registro de la prioridad de la Demo (escrito ANTES de evaluar; 2026-11-22, ~14:20)

Comentario recibido en la Demo (20-11): el "auto de adelante" puede ir en **otra vuelta** (un auto doblado o uno que nos dobló). En circuitos cortos los doblajes son más frecuentes, y un gap a un auto que va 2 vueltas atrás no representa tráfico de carrera. Hay que **limitar el gap a autos en la misma vuelta**. Coincide con el riesgo (i) declarado en HITO1 §3.

| Elemento | Definición fijada antes de mirar resultados |
|---|---|
| Feature | `gap_samelap_s(t)` = `end_time_s(t)` − `end_time_s` del último auto (otro piloto, misma carrera) que cruzó la línea **completando el mismo número de vuelta** que t, con `end_time_s ≤ end_time_s(t)`. Representación: `gap_samelap_cap_s = min(gap_samelap_s, 5 s)` |
| Faltantes | Si nadie completó esa vuelta antes (el piloto va primero en esa vuelta): tope 5 s ("aire libre"), igual que la regla del Hito 2. Se reporta cuántas filas la usan |
| Diagnóstico previo (sin target) | Proporción de pares en que el auto usado por `gap_ahead_s` (Hito 2) iba en otra vuelta, por carrera |
| Modelo | **E4** = OLS 6 inputs + `gap_samelap_cap_s`. Una sola variante (sin búsqueda); se reporta su MAE LORO en R1–R7 junto a E0 y E1 |
| Control pareado | **E0** (mismo OLS sin la feature). Se reporta también contra E1 para aislar el efecto de restringir a la misma vuelta |
| Reporte | MAE total y por carrera con n, control − candidato, IC 95 % bootstrap por bloques piloto-carrera, tramo < 1 s |
| Apoya / contradice | Mismos criterios que el pre-registro anterior: apoya solo si mejora a E0 global, en R8 y R9, y en el tramo < 1 s |

## Resultados y decisiones · informe final

| Fecha | Quién | Decisión | Evidencia | Siguiente paso |
|---|---|---|---|---|
| 2026-11-22 | Valentín | Selección del cambio del gap en R1–R7: **E3a** | LORO: E0 0,50839 · E1 0,50858 · E3a 0,50919 · E3b 0,50939 · E4 0,50916 s (`salidas_final/selection_loro_train_final.csv`). E3b no supera a E3a por > 0,005 s. Ya en train ninguna representación mejora a E0 (trials no exitosos) | — |
| 2026-11-22 | Valentín | **Descartar E3a** (cambio del gap) | E0 − E3a = −0,208 ms, IC 95 % [−1,32 ; +0,88] ms; R8 −0,26, R9 −0,13 ms; tramo < 1 s −0,65 ms (`control_vs_candidate_by_race_final.csv`, `mae_by_gap_bin_final.csv`) | — |
| 2026-11-22 | Valentín | **Descartar E4** (gap en la misma vuelta, prioridad de la Demo) | Diagnóstico: el auto de E1 iba en otra vuelta en el 32,9 % de validación. E0 − E4 = +0,006 ms, IC [−0,39 ; +0,40] ms; R8 −0,37, R9 +0,55 ms; tramo < 1 s −1,21 ms. No cumple el criterio "apoya". E1 − E4 = +0,348 ms (`run_final.json`) | Un modelo no lineal o por circuito con el gap en la misma vuelta, con el mismo protocolo |
| 2026-11-22 | Valentín | Acotar el placebo del Hito 2 a la permutación observada y repetirlo 200 veces | Percentil 48,5; E1 peor que el 77 % de los placebos (`placebo_E1_permutations.csv`) | — |
| 2026-11-22 | Equipo | **Retener E0** como modelo final y **no nominar** candidato al bono | Ningún candidato cumple los criterios pre-registrados; la ganancia de E4 (+0,006 ms) está dentro del ruido | — |

## Uso de IA · informe final

| Quién | Herramienta | Para qué | Qué se adoptó / cambió | Cómo se verificó |
|---|---|---|---|---|
| Valentín | Claude | Leer el enunciado del informe final y el feedback del Hito 2 en Canvas, proponer el reparto, escribir `final_experiment.py`, el pre-registro y el borrador de FINAL_REPORT §1, §2, §5, §6. Prompt representativo: «ahora viene esto así que divídelo en dos igual y trabaja» | Estructura del script, checks F0–F10, placebo con 200 permutaciones. La representación E4 viene del comentario de la Demo que yo transmití; E3a, del feedback del profe | 23 checks OK; E0/E1 reproducen el Hito 2 (F5) y la semilla 414 el placebo del Hito 2 (F10); MAE global = R8/R9 ponderado por n (F9); cifras de §2 y §6 contrastadas con `salidas_final/*.csv` |
| Camilo | *(completar)* | | | |
