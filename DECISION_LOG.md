# DECISION_LOG · Hito 1 NextLap

| Fecha | Quién | Decisión | Evidencia | Cambio posterior / siguiente paso |
|---|---|---|---|---|
| 2026-11-11 | Equipo | Reparto del trabajo en dos mitades (ver REPARTO.md) | Rúbrica Hito 1 v2 | |
| 2026-11-11 | Valentín | Script propio `hito1_reference.py` que reutiliza el helper del curso en vez de entregar solo `probar_referencia_v1.py` | La rúbrica pide reproducción propia; la salida del instructor no cuenta | Se agregaron 9 checks (SHA-256, conteos, rondas, `pair_id` disjuntos, orden temporal, solo clean, sin `next_*`, scaler ajustado solo en train, mismas filas) |
| 2026-11-11 | Valentín | Reportar MAE por carrera y comparación pareada OLS vs persistencia | `salidas_hito1/paired_ols_vs_persistence.csv`: OLS −0,023 s, pero gana solo en el 51,6 % de las filas | La mejora de la referencia es modesta: cualquier extensión del Hito 2 debe compararse contra OLS en las mismas filas y por carrera |
| 2026-11-11 | Valentín | Guardar el hash del set de `pair_id` de validación (`f453a98d…`) | `reference_run.json` | En el Hito 2, verificar que la extensión y la ablación usen exactamente ese set |
| 2026-11-11 | Valentín | Cuantificar el límite de `clean_pair` con `observed_pairs_train_validation_v1.csv` | 82,4 % (R8) y 71,8 % (R9) de los pares observados son clean | Reportar por separado cualquier análisis sobre la población amplia |
| | | | | |

## Revisión tras feedback

- Entrada original:
- Feedback recibido (de quién):
- Entrada revisada o razón para mantenerla:

## Uso de IA

| Quién | Herramienta | Para qué | Sugerencia adoptada | Cómo se verificó |
|---|---|---|---|---|
| Valentín | Claude | Leer el enunciado, descargar materiales, proponer reparto y plantillas | Reparto por criterio de rúbrica; plantillas HITO1/DECISION_LOG | Contrastado con el brief y la rúbrica v2 de Canvas |
| Valentín | Claude | Redactar `hito1_reference.py`, README y borrador de las secciones 1–2 de HITO1.md | Script con checks, comparación pareada y texto del encuadre | Ejecuté el script en mi computador (Python 3.12, numpy 2.5.4, pandas 3.1.1, sklearn 1.9.1): 9 checks OK y MAE iguales a `salidas_referencia/` del instructor (0,3867 / 0,4093 / 11,3761 s). Revisé cada cifra del texto contra `salidas_hito1/` y los porcentajes de `clean_pair` contra `observed_pairs_train_validation_v1.csv` |
| | | | | |

## Contribuciones reales

- Valentín: encuadre del problema y límite de `clean_pair` (HITO1 §1); reproducción de las tres referencias con checks y la interpretación (HITO1 §2, `hito1_reference.py`, `salidas_hito1/`); README, entorno y empaquetado final.
- ________:

## Antes del Hito 2

- Responsable / tarea:
- Bloqueo conocido:
