"""Hito 2 · NextLap — errores por contexto (§2) y SHAP (§3) · parte Camilo (C4, C5, C6).

Requiere haber corrido antes, en el MISMO entorno:  python hito2_experiment.py
(los .joblib dependen de la versión de scikit-learn; ver salidas_hito2/run.json → environment).

Qué hace, en orden:
  1. Carga E0 (referencia OLS, el modelo que retenemos) y E1 (candidato) desde salidas_hito2/models/
     y verifica que reproducen las predicciones de validation_predictions.csv.
  2. §2 · Caso real por pair_id (CASE_PAIR): observado, predicciones y |error| recalculados aquí,
     y el desplazamiento del candidato vs control por tramo de gap (signo del error de E0).
  3. §3 · SHAP con shap.LinearExplainer sobre la regresión ya preprocesada:
       - fondo  = BG_N filas de train_features.csv (solo R1–7), seed 414, transformadas con
                  model.named_steps['prepare'] (ajustado solo con train).
       - local  = CASE_PAIR: base + suma de contribuciones = predicción (tolerancia 1e-6).
       - global = media |SHAP| por input original sobre las 1.808 filas de validación (R8–9).
     Las columnas codificadas numeric__* / category__* se suman de vuelta a su input original
     (en un modelo lineal la suma conserva la aditividad).
     Check extra: para OLS con fondo independiente, SHAP_j = coef_j · (x_j − media_fondo_j); se
     recalcula a mano y se compara con shap.
  4. Guarda salidas_hito2/shap_*.csv y shap_run.json.

Uso (desde la raíz del repo):  python hito2_shap.py
"""
from pathlib import Path
import json, platform

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'salidas_hito2'
RANDOM_SEED = 414
BG_N = 1000                      # tamaño del fondo (muestra de train)
CASE_PAIR = '2026-R8-3-L10'      # gap 0,644 s, error grande del candidato y candidato peor que control
TOL = 1e-6
MODELS = {'E0_reference_ols': 'E0 referencia (retenido)', 'E1_candidate': 'E1 candidato (+gap)'}

checks = []


def check(name, ok, detail=''):
    checks.append({'check': name, 'passed': bool(ok), 'detail': str(detail)})
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def original_input(encoded, numeric, categorical):
    """numeric__current_lap_s -> current_lap_s ; category__current_team_McLaren -> current_team."""
    kind, rest = encoded.split('__', 1)
    if kind == 'numeric':
        return rest
    return next(c for c in categorical if rest.startswith(c + '_'))


# =============================================================== 1. datos y modelos
train = pd.read_csv(OUT / 'train_features.csv')
val = pd.read_csv(OUT / 'validation_features.csv')
preds = pd.read_csv(OUT / 'validation_predictions.csv')
check('S0 fondo solo de train (R1–7) y validación R8–9',
      set(train['round']) == set(range(1, 8)) and set(val['round']) == {8, 9} and len(val) == 1808,
      f'train={sorted(set(train["round"]))} val={sorted(set(val["round"]))} n_val={len(val)}')
check('S0b mismas filas y orden que validation_predictions.csv', val.pair_id.equals(preds.pair_id), 'pair_id')

bundles = {run: joblib.load(OUT / 'models' / f'{run}.joblib') for run in MODELS}
for run, b in bundles.items():
    p = b['model'].predict(val[b['numeric'] + b['categorical']])
    diff = np.max(np.abs(p - preds[f'pred_{run}']))
    check(f'S1 {run}.joblib reproduce pred_{run}', diff < TOL, f'máx |dif| = {diff:.2e} s')

# =============================================================== 2. §2 caso real y contexto
row = preds.loc[preds.pair_id == CASE_PAIR].iloc[0]
case = pd.DataFrame([{
    'pair_id': CASE_PAIR, 'event_name': row.event_name, 'driver_number': row.driver_number,
    'current_lap_number': row.current_lap_number, 'current_lap_s': row.current_lap_s,
    'gap_ahead_s': row.gap_ahead_s, 'y_true_s': row.y_true_s,
    'pred_E0_reference_ols': row.pred_E0_reference_ols, 'pred_E1_candidate': row.pred_E1_candidate,
    'abs_err_E0_hand': abs(row.y_true_s - row.pred_E0_reference_ols),
    'abs_err_E1_hand': abs(row.y_true_s - row.pred_E1_candidate),
}])
case['control_minus_candidate_s'] = case.abs_err_E0_hand - case.abs_err_E1_hand
check('S2 |error| a mano = abs_err_* publicado',
      np.isclose(case.abs_err_E0_hand.iat[0], row.abs_err_E0_reference_ols, atol=TOL)
      and np.isclose(case.abs_err_E1_hand.iat[0], row.abs_err_E1_candidate, atol=TOL), CASE_PAIR)
case.to_csv(OUT / 'shap_case_error.csv', index=False)

# desplazamiento del candidato (E1 − E0) por tramo y si la vuelta real fue más lenta que lo predicho por E0
ctx = preds.assign(shift_E1_minus_E0_s=preds.pred_E1_candidate - preds.pred_E0_reference_ols,
                   obs_slower_than_E0=preds.y_true_s > preds.pred_E0_reference_ols,
                   d=preds.abs_err_E0_reference_ols - preds.abs_err_E1_candidate)
order = ['<1s', '1-2s', '2-5s', '>=5s']
context = (ctx.groupby('gap_bin')
           .apply(lambda g: pd.Series({
               'n': len(g),
               'mean_shift_E1_minus_E0_s': g.shift_E1_minus_E0_s.mean(),
               'share_obs_slower_than_E0': g.obs_slower_than_E0.mean(),
               'control_minus_candidate_s': g.d.mean(),
               'control_minus_candidate_when_slower_s': g.loc[g.obs_slower_than_E0, 'd'].mean(),
               'control_minus_candidate_when_faster_s': g.loc[~g.obs_slower_than_E0, 'd'].mean()}),
                  include_groups=False)
           .reindex(order).reset_index())
context.to_csv(OUT / 'shap_context_by_gap_bin.csv', index=False)

# =============================================================== 3. SHAP
bg_raw = train.sample(n=BG_N, random_state=RANDOM_SEED)
local_rows, global_rows, meta = [], [], {}
case_idx = int(np.flatnonzero(val.pair_id == CASE_PAIR)[0])

for run, label in MODELS.items():
    b = bundles[run]
    cols, cats = b['numeric'] + b['categorical'], b['categorical']
    prepare, reg = b['model'].named_steps['prepare'], b['model'].named_steps['regression']
    names = list(prepare.get_feature_names_out())
    X_bg = prepare.transform(bg_raw[cols])
    X_val = prepare.transform(val[cols])

    explainer = shap.LinearExplainer(reg, shap.maskers.Independent(X_bg, max_samples=BG_N))
    sv = explainer(X_val)
    vals, base = sv.values, float(np.ravel(sv.base_values)[0])
    pred = b['model'].predict(val[cols])

    add_err = np.max(np.abs(base + vals.sum(axis=1) - pred))
    check(f'S3 {run}: base + suma SHAP = predicción (1.808 filas)', add_err < TOL, f'máx |dif| = {add_err:.2e} s')
    hand = (X_val - X_bg.mean(axis=0)) * reg.coef_
    hand_err = np.max(np.abs(hand - vals))
    check(f'S4 {run}: SHAP = coef·(x − media fondo) recalculado a mano', hand_err < TOL, f'máx |dif| = {hand_err:.2e}')
    base_err = abs(base - reg.predict(X_bg).mean())
    check(f'S5 {run}: base value = predicción media del fondo', base_err < TOL, f'|dif| = {base_err:.2e} s')

    # agregar columnas codificadas -> input original
    groups = pd.Series([original_input(n, b['numeric'], cats) for n in names])
    agg = pd.DataFrame(vals, columns=names).T.groupby(groups.values, sort=False).sum().T

    # local
    x = val.iloc[case_idx]
    for inp in agg.columns:
        local_rows.append({'model': run, 'pair_id': CASE_PAIR, 'input': inp, 'value': x[inp],
                           'shap_s': agg.iloc[case_idx][inp]})
    total = agg.iloc[case_idx].sum()
    local_rows += [
        {'model': run, 'pair_id': CASE_PAIR, 'input': '[base value]', 'value': '', 'shap_s': base},
        {'model': run, 'pair_id': CASE_PAIR, 'input': '[base + suma]', 'value': '', 'shap_s': base + total},
        {'model': run, 'pair_id': CASE_PAIR, 'input': '[predicción modelo]', 'value': '', 'shap_s': pred[case_idx]},
        {'model': run, 'pair_id': CASE_PAIR, 'input': '[observado y_true_s]', 'value': '', 'shap_s': x['next_lap_s']},
        {'model': run, 'pair_id': CASE_PAIR, 'input': '[|error| aparte]', 'value': '',
         'shap_s': abs(x['next_lap_s'] - pred[case_idx])},
    ]
    check(f'S6 {run}: caso {CASE_PAIR} base + suma = predicción', abs(base + total - pred[case_idx]) < TOL,
          f'{base + total:.6f} vs {pred[case_idx]:.6f}')

    # global
    g = agg.abs().mean().sort_values(ascending=False)
    for rank, (inp, v) in enumerate(g.items(), 1):
        global_rows.append({'model': run, 'rank': rank, 'input': inp, 'mean_abs_shap_s': v,
                            'share_of_total': v / g.sum()})
    meta[run] = {'label': label, 'base_value_s': base, 'n_encoded_columns': len(names)}

pd.DataFrame(local_rows).to_csv(OUT / 'shap_local.csv', index=False)
pd.DataFrame(global_rows).to_csv(OUT / 'shap_global.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'shap_checks.csv', index=False)

json.dump({
    'explainer': 'shap.LinearExplainer (masker Independent = interventional) sobre la regresión OLS ya preprocesada',
    'background': f'{BG_N} filas de train_features.csv (R1–7), sample(random_state={RANDOM_SEED}), '
                  "transformadas con model.named_steps['prepare'] (ajustado solo con train)",
    'local_case': CASE_PAIR,
    'global_sample': 'las 1.808 filas de validación R8–9 (validation_features.csv), sin muestreo',
    'aggregation': 'columnas numeric__*/category__* sumadas a su input original; global = media |SHAP| por input',
    'units': 'segundos de next_lap_s',
    'models': meta,
    'checks_passed': len(checks),
    'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                    'sklearn': sklearn.__version__, 'shap': shap.__version__, 'platform': platform.platform()},
}, open(OUT / 'shap_run.json', 'w'), indent=2, ensure_ascii=False)

# =============================================================== resumen en consola
pd.set_option('display.width', 160)
print('§2 caso:\n', case.T.to_string(header=False), '\n')
print('§2 contexto por tramo de gap:\n', context.round(4).to_string(index=False), '\n')
print('§3 SHAP local:\n', pd.DataFrame(local_rows).round(6).to_string(index=False), '\n')
print('§3 SHAP global:\n', pd.DataFrame(global_rows).round(4).to_string(index=False), '\n')
print(f'{len(checks)} checks OK')
