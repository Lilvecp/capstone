# Hito 2 NextLap · Reparto (entrega jue 19 nov, 23:59)

## Resultado ya ejecutado (Valentín)
Resultado **negativo** válido: el gap al auto de adelante no mejora el MAE.
Control (= OLS ref) 0,38671 s · Candidato 0,38705 s · control − candidato = **−0,00034 s** (IC 95 % [−0,0027; +0,0021]).
Por carrera: R8 −0,0019 s (n=1.066) · R9 +0,0019 s (n=742). Decisión: descartar la feature y retener la referencia.
Todo sale de `python hito2_experiment.py` → `salidas_hito2/`.

## Valentín · hecho (C1, C2, C3, C7 ≈ 55 pts)
`hito2_experiment.py`, `salidas_hito2/`, HITO2 §0, §1, §4, README §Hito 2, DECISION_LOG §Hito 2, su párrafo en §5.

## Camilo · pendiente (C4, C5, C6 ≈ 40 pts + su párrafo)

### §2 Errores por contexto (C4, 20 pts)
1. Tabla R8/R9 desde `salidas_hito2/control_vs_candidate_by_race.csv`: n, MAE control, MAE candidato, control − candidato. Decir si el patrón coincide con el global: R8 favorece al control, R9 al candidato, y el global (ponderado por n: (1066·d8 + 742·d9)/1808) queda casi en 0 → **no** es una mejora consistente.
2. Elegir **un `pair_id` real** en `validation_predictions.csv` (sugerencia: uno con `gap_ahead_s < 1` y error grande del candidato). Anotar `y_true_s`, `pred_E1_candidate` (y `pred_E0_reference_ols`), y calcular |error| a mano.
3. Limitación concreta + siguiente check con esa evidencia (ej.: en gap < 1 s el candidato empeora 0,0059 s, n=328 → el gap en la línea no dice si el auto sigue cerca en t+1; siguiente check: gap medio de las últimas 2–3 vueltas, o si el auto de adelante es de la misma vuelta).

### §3 SHAP (C5 10 pts + C6 10 pts)
- Modelo: **E0 (referencia)**, porque es el que retenemos según la ablación (o E1 si quieren mostrar cuánto pesa el gap; decir por qué).
- Corre primero `python hito2_experiment.py` en TU entorno (los `.joblib` dependen de la versión de sklearn).
- Explainer: `shap.LinearExplainer` sobre el modelo ya preprocesado, fondo = muestra de `salidas_hito2/train_features.csv` (solo train, seed 414) transformada con `model.named_steps['prepare']`. Contribuciones en segundos. Mapeo `numeric__*` / `category__*` → input original (`candidate_coefficients.csv`).
- Local: un `pair_id` (puede ser el mismo de §2): predicción, base value, contribuciones con signo, **check base + suma = predicción** (tolerancia ej. 1e-6), |error| aparte. Interpretar el signo respecto de la base, sin hablar de causalidad.
- Global: media |SHAP| por input sobre una muestra declarada (ej. las 1.808 filas de validación o 500 con seed 414), guardar CSV/plot en `salidas_hito2/shap_*`. Decir explainer, fondo, muestra, cómo se agregó, y el límite (solo esa muestra / R8–R9, asociación del modelo, no prueba de menor error).
- Guardar script `hito2_shap.py` y agregarlo a la tabla del README.

### Compartido
- Párrafo individual en HITO2 §5 (decisión propia + resultado que la respalda + limitación/siguiente check + IA y verificación).
- Fila propia en DECISION_LOG §Hito 2 (Uso de IA y Contribuciones).

## Cierre (jue 19)
Valentín: revisar §2–3 de Camilo, commit + tag `Hito2`, `Hito2_Castro-Bufadel.zip`, subir a Canvas con URL y commit.
