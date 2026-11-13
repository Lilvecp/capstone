# DECISION_LOG · Hito 1 NextLap

| Fecha | Quién | Decisión | Evidencia | Cambio posterior / siguiente paso |
|---|---|---|---|---|
| 2026-11-11 | Equipo | Reparto del trabajo en dos mitades (ver REPARTO.md) | Rúbrica Hito 1 v2 | |
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
