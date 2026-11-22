# Informe final NextLap · Reparto (entrega dom 22 nov, 23:59 · un PDF por equipo)

## Ya hecho (Valentín)
`final_experiment.py` → `salidas_final/` (23 checks). FINAL_REPORT §1, §2, §5, §6, README, DECISION_LOG (pre-registro, resultados, IA).
Resultado: ninguna representación del tráfico mejora a E0. E3a (cambio del gap) −0,208 ms; E4 (misma vuelta, pedido de la Demo) +0,006 ms con IC centrado en 0. **Se retiene E0.**

## Camilo · pendiente (C3 15 pts + C4 15 pts)
Correr primero: `python final_experiment.py` (después de hito2_experiment.py).

### §3 Errores por contexto
1. Tabla R8/R9 (n, MAE control, MAE candidato, control − candidato) para E0 vs E4 y E0 vs E1, desde `salidas_final/control_vs_candidate_by_race_final.csv`; explicar la relación con el global (ponderado por n).
2. Caso inspeccionado: puede seguir siendo `2026-R8-3-L10` (ahí el auto de adelante SÍ iba en la misma vuelta: gap 0,644 s; E4 predice 72,699; E0 72,708; observado 74,413). Verificar |error| a mano. Ojo: también sirve buscar un caso donde el auto de adelante iba en otra vuelta (`ahead_other_lap == True` en `validation_predictions_final.csv`) para mostrar el problema de la Demo.
3. Limitación observada + siguiente check concreto (ej.: en < 1 s ningún candidato mejora; siguiente check: modelo con interacción gap × circuito o no lineal, mismo protocolo).

### §4 SHAP
- Mantener E0 (modelo retenido). Local con base + contribuciones = predicción y |error| aparte.
- **Corrección del profe:** en Austria, el SHAP del gap en E1 (−50,9 ms) es respecto de la base de E1; el cambio de predicción E1 − E0 es −44,5 ms (72,663 − 72,708) e incluye el reajuste del modelo. Reescribir esa frase del Hito 2 (decía "es lo que baja la predicción").
- Global: muestra, explainer, fondo de train, alcance (ya está en `shap_run.json`).

### Además
- Fila de Camilo en "Uso de IA · informe final" (DECISION_LOG y FINAL_REPORT §5c) **con prompt representativo**.
- Tu decisión del informe final en §5b y §7.
- `git pull`, editar, `git push`. Valentín arma el PDF y entrega.
