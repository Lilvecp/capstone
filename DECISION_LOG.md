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
| 2026-11-18 | Valentín | S1 implementado con el auto de adelante que no está en vuelta de pit in/out | MAE S1 0,38731 s: no cambia la conclusión | Duda de Hito 1 sobre autos en pit lane queda resuelta para este resultado |

## Revisión tras feedback (Hito 1 → Hito 2)

- Feedback C1.3 (instructor, Hito 1): "clean_pair does not independently exclude strong traffic".
- Entrada original (HITO1 §1): "Las vueltas con safety car, pit o tráfico fuerte quedan fuera".
- Verificación: el 18,1 % de los pares de validación (328/1.808) y el 22,9 % de train tienen un auto a < 1 s (`salidas_hito2/feature_coverage.csv`, `mae_by_gap_bin.csv`).
- Cambio: la afirmación se corrige en HITO2 §0 (y en el informe final): `clean_pair` excluye pit, banderas, vueltas no precisas/borradas y cambio de stint, **no** tráfico. El análisis por tramo de gap se mantiene (HITO2 §1).

## Uso de IA · Hito 2

| Quién | Herramienta | Para qué | Qué se adoptó / cambió | Cómo se verificó |
|---|---|---|---|---|
| Valentín | Claude | Utilicé Claude como apoyo para organizar mis ideas, revisar los requisitos del Hito 2 y facilitar la implementación del código y la redacción. Yo definí cómo dividir el trabajo con Camilo, qué experimentos realizar y cómo presentar los resultados.
Decidí mantener el OLS de referencia como control y evaluar el aporte de la variable de tráfico. Con apoyo de Claude implementé el script hito2_experiment.py, los checks K0–K9 y el bootstrap por bloques. Mantuve los criterios establecidos en el Hito 1, incluyendo la feature, el tope de 5 s y la regla de selección.Ejecuté el script en mi computador y comprobé que E0 reprodujera el MAE del Hito 1 (0,386707 s, K7). También verifiqué el MAE global mediante el promedio ponderado de R8 y R9 (K9), y contrasté las cifras de HITO2 §1 con los archivos CSV generados.|
| Camilo | *(completar)* | | | |

## Contribuciones reales · Hito 2

- Valentín: `hito2_experiment.py` (feature, selección LORO, E0/E1/E2/S1, checks K0–K9, predicciones por `pair_id`, bootstrap, tramos de gap); HITO2 §0, §1, §4; README Hito 2; corrección C1.3. Puede explicar: por qué E0 es el control pareado, por qué la diferencia negativa descarta la feature y por qué K3 se revisó.
- Camilo: *(completar: HITO2 §2 errores por contexto, §3 SHAP, script/salidas SHAP)*.
