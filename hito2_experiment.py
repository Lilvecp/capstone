"""Hito 2 · NextLap — experimento ejecutado (parte Valentín: C1, C2, C3, C7).

Hipótesis (HITO1 §3): la proximidad del auto de adelante al final de la vuelta t
(`gap_ahead_s`, calculado con `end_time_s` de los snapshots publicados) agrega información
sobre la vuelta t+1 que los 6 inputs de la referencia no tienen.

Qué hace, en orden:
  1. Verifica datos (SHA-256), split R1–7 / R8–9, conteos y unicidad de `pair_id`.
  2. Construye la feature sobre la línea de tiempo ORIGINAL completa (todas las vueltas de todos
     los pilotos, antes del filtro retrospectivo `clean_pair`) usando solo autos que cruzaron la
     línea con `end_time_s <= end_time_s(t)`. La une a los CSV publicados por `pair_id`.
  3. Selección pre-declarada (HITO1 §4): E1 (gap con tope 5 s) vs E1b (indicador gap < 1 s) con
     leave-one-race-out DENTRO de train (R1–7). Validación no participa en la selección.
  4. Validación R8–9, una sola vez: E0 referencia OLS (= control pareado, mismo OLS sin la feature),
     E1 candidato, E2 placebo (feature permutada dentro de cada carrera, seed 414) y S1 sensibilidad
     (sin contar como "auto adelante" a los que están en vuelta de pit in/out).
  5. Guarda predicciones por `pair_id`, tablas de MAE (global y por carrera), diferencia pareada con
     IC bootstrap por bloques piloto-carrera, MAE por tramo de gap, trials, checks K1–K8, modelos
     ajustados (joblib, para SHAP) y entorno.

Reutiliza con atribución `load_snapshots`, `build_pairs`, `make_model`, `NUMERIC`, `CATEGORICAL`,
`TARGET` de `base_NextLap/nextlap_reference_v1.py` (helper del curso IIT414W).

Uso (desde la raíz del repo):  python hito2_experiment.py   → salidas_hito2/
"""
from pathlib import Path
import hashlib, json, platform, sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'base_NextLap'
OUT = ROOT / 'salidas_hito2'
sys.path.insert(0, str(BASE))
from nextlap_reference_v1 import (load_snapshots, build_pairs, make_model,  # noqa: E402
                                  NUMERIC, CATEGORICAL, TARGET)

RANDOM_SEED = 414
GAP_CAP_S = 5.0                 # pre-declarado en HITO1 §3
DRS_THRESHOLD_S = 1.0           # alternativa pre-declarada E1b
TIE_MARGIN_S = 0.005            # regla de selección pre-declarada en HITO1 §4
VAL_HASH = 'f453a98dc906904c'   # hash del set de pair_id de validación (Hito 1, reference_run.json)
HITO1_OLS_MAE = 0.3867          # MAE OLS referencia reproducido en Hito 1 (salidas_hito1/reference_metrics.csv)
TRAIN_ROUNDS, VAL_ROUNDS = list(range(1, 8)), [8, 9]
EXPECTED_SHA = {
    'train_clean_v1.csv': '43544f38cc9ef32b44886e6751e03522601265160086fe3a4956f04b87bd5c13',
    'validation_clean_v1.csv': 'b19efb6a908af12a5d2c6bdde29aa40984295e62ce0f187a201b579278db7ee8',
    'nextlap_reference_v1.py': 'f6c58925cd4a477bf3e1a5a2b3968e67df3a1cffabecee9e89701c3ac2d5f994',
}

checks, trials = [], []


def check(name, ok, detail=''):
    checks.append({'check': name, 'passed': bool(ok), 'detail': str(detail)})
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


# =============================================================== 1. datos y split
hashes = {n: sha256(BASE / n) for n in EXPECTED_SHA}
check('K0a versión de datos (SHA-256)', hashes == EXPECTED_SHA, {k: v[:12] for k, v in hashes.items()})

train = pd.read_csv(BASE / 'train_clean_v1.csv', dtype={'driver_number': str})
val = pd.read_csv(BASE / 'validation_clean_v1.csv', dtype={'driver_number': str})
train_rounds, val_rounds = [int(r) for r in sorted(train['round'].unique())], [int(r) for r in sorted(val['round'].unique())]
check('K0b split por carrera: train R1–7, validación R8–9',
      train_rounds == TRAIN_ROUNDS and val_rounds == VAL_ROUNDS,
      f'train={train_rounds} val={val_rounds}')
check('K0c conteos publicados', (len(train), len(val), (val['round'] == 8).sum(), (val['round'] == 9).sum())
      == (5772, 1808, 1066, 742), f'train={len(train)} val={len(val)}')
check('K0d pair_id único y disjunto', train.pair_id.is_unique and val.pair_id.is_unique
      and not set(train.pair_id) & set(val.pair_id), 'train ∩ val = ∅')

# =============================================================== 2. feature en la línea de tiempo original
raw, manifest = load_snapshots(BASE)          # verifica SHA-256 de cada snapshot contra el manifest
raw['round'] = raw['round'].astype(int)
check('K0e snapshots sin test', set(raw['round']) == set(range(1, 10)), sorted(set(raw['round'])))

laps = raw.dropna(subset=['end_time_s']).sort_values(['round', 'end_time_s', 'driver_number']).copy()
laps = laps.reset_index(drop=True)
# Principal: auto (otro piloto) que cruzó la línea inmediatamente antes que este piloto terminara t.
laps['ahead_end_time_s'] = laps.groupby('round').end_time_s.shift(1)
laps['ahead_driver'] = laps.groupby('round').driver_number.shift(1)
laps['gap_ahead_s'] = laps.end_time_s - laps.ahead_end_time_s

# S1: mismo cálculo, pero el "auto de adelante" no puede estar en vuelta de pit in / pit out.
pit_flag = (laps.pit_in.astype(bool) | laps.pit_out.astype(bool)).to_numpy()
s1_t, s1_d = np.full(len(laps), np.nan), np.full(len(laps), None, dtype=object)
for _, idx in laps.groupby('round').indices.items():      # idx ya está ordenado por end_time_s
    last = []                                              # últimos cruces de autos fuera de pit, en orden
    for i in idx:
        drv = laps.driver_number.iat[i]
        prev = next(((t, d) for t, d in reversed(last) if d != drv), None)   # otro piloto, ya cruzó
        if prev is not None:
            s1_t[i], s1_d[i] = prev
        if not pit_flag[i]:
            last.append((laps.end_time_s.iat[i], drv))
            last = last[-30:]
laps['ahead_nopit_end_time_s'] = s1_t
laps['ahead_nopit_driver'] = s1_d
laps['gap_ahead_nopit_s'] = laps.end_time_s - laps.ahead_nopit_end_time_s

m = laps.ahead_end_time_s.notna()
check('K1 temporal: el auto usado cruzó antes o junto con el fin de t',
      (laps.ahead_end_time_s[m] <= laps.end_time_s[m]).all()
      and (laps.ahead_nopit_end_time_s.dropna() <= laps.end_time_s[laps.ahead_nopit_end_time_s.notna()]).all(),
      f'max(end_time_ahead - end_time_t) = {(laps.ahead_end_time_s - laps.end_time_s).max():.3f} s (≤ 0)')
check('K1b el auto usado nunca es el mismo piloto',
      (laps.ahead_driver[m] != laps.driver_number[m]).all()
      and (laps.ahead_nopit_driver.dropna() != laps.driver_number[laps.ahead_nopit_driver.notna()]).all(), '')

pairs = build_pairs(raw)
pairs['round'] = pairs['round'].astype(int)
feat_cols = ['gap_ahead_s', 'gap_ahead_nopit_s', 'ahead_end_time_s', 'ahead_driver']
pairs = pairs.merge(laps[['round', 'driver_number', 'lap_number'] + feat_cols]
                    .rename(columns={'lap_number': 'current_lap_number'}),
                    on=['round', 'driver_number', 'current_lap_number'], how='left', validate='one_to_one')

# La feature se calcula sin usar clean_pair. Diagnóstico: cuántas filas cambiarían si se filtrara primero.
clean_laps = laps.merge(pairs.loc[pairs.clean_pair, ['round', 'driver_number', 'current_lap_number']]
                        .rename(columns={'current_lap_number': 'lap_number'}), on=['round', 'driver_number', 'lap_number'])
clean_laps = clean_laps.sort_values(['round', 'end_time_s', 'driver_number'])
clean_laps['gap_if_filtered_first'] = clean_laps.end_time_s - clean_laps.groupby('round').end_time_s.shift(1)
n_would_change = int((~np.isclose(clean_laps.gap_if_filtered_first.fillna(-1), clean_laps.gap_ahead_s.fillna(-1))).sum())

pairs['end_time_s_rebuilt'] = pairs.end_time_s
join_cols = ['pair_id', 'end_time_s_rebuilt', 'gap_ahead_s', 'gap_ahead_nopit_s', 'ahead_end_time_s', 'ahead_driver']
before = (len(train), len(val))
train = train.merge(pairs[join_cols], on='pair_id', how='left', validate='one_to_one')
val = val.merge(pairs[join_cols], on='pair_id', how='left', validate='one_to_one')
check('K2a join por pair_id sin filas perdidas ni duplicadas',
      (len(train), len(val)) == before and train.pair_id.is_unique and val.pair_id.is_unique,
      f'train {before[0]}→{len(train)}, val {before[1]}→{len(val)}')
check('K2b mismo set de validación que Hito 1 (hash)', set_hash(val.pair_id).startswith(VAL_HASH),
      set_hash(val.pair_id)[:16])
check('K3a sin fuga: inputs sin next_* ni clean_pair', True,
      'make_model bloquea next_*; clean_pair nunca entra en las listas de inputs (ver abajo)')
check('K3b la feature se construyó antes del filtro clean_pair', n_would_change > 0,
      f'{n_would_change} pares clean tendrían otro gap si se filtrara primero; aquí se usa la línea completa')
check('K3c fin de t publicado = reconstruido, y auto usado ≤ fin de t',
      np.allclose(val.end_time_s, val.end_time_s_rebuilt) and np.allclose(train.end_time_s, train.end_time_s_rebuilt) and val.ahead_end_time_s.le(val.end_time_s).all()
      and train.ahead_end_time_s.le(train.end_time_s).all(), 'último timestamp usado ≤ fin de t en cada par')

# Regla de faltantes (pre-declarada): si no hay auto adelante → tope (5 s = aire libre).
coverage = []
for name, d in [('train R1-7', train), ('validation R8-9', val)]:
    coverage.append({'population': name, 'n': len(d),
                     'gap_missing': int(d.gap_ahead_s.isna().sum()),
                     'gap_nopit_missing': int(d.gap_ahead_nopit_s.isna().sum()),
                     'coverage_gap': float(d.gap_ahead_s.notna().mean()),
                     'share_lt_1s': float((d.gap_ahead_s < 1).mean()),
                     'share_ge_cap': float((d.gap_ahead_s >= GAP_CAP_S).mean())})
    for c in ['gap_ahead_s', 'gap_ahead_nopit_s']:
        d[c + '_filled'] = d[c].fillna(GAP_CAP_S)
coverage = pd.DataFrame(coverage)
check('K5 faltantes contados (regla: tope 5 s)', coverage.loc[1, 'gap_missing'] == 0,
      coverage[['population', 'gap_missing', 'gap_nopit_missing']].to_dict('records'))

for d in (train, val):
    d['gap_ahead_cap_s'] = d.gap_ahead_s_filled.clip(upper=GAP_CAP_S)
    d['gap_lt_1s'] = (d.gap_ahead_s_filled < DRS_THRESHOLD_S).astype(float)
    d['gap_ahead_nopit_cap_s'] = d.gap_ahead_nopit_s_filled.clip(upper=GAP_CAP_S)
    d['gap_nopit_lt_1s'] = (d.gap_ahead_nopit_s_filled < DRS_THRESHOLD_S).astype(float)


def placebo(d, col):
    """Permuta la feature dentro de cada carrera (seed 414): misma distribución marginal, sin vínculo con la fila."""
    rng = np.random.default_rng(RANDOM_SEED)
    out = d[col].copy()
    for r in sorted(d['round'].unique()):
        idx = d.index[d['round'] == r]
        out.loc[idx] = rng.permutation(d.loc[idx, col].to_numpy())
    return out


# =============================================================== 3. selección con LORO dentro de train
CATS = CATEGORICAL


def fit(frame, numeric):
    for c in numeric + CATS:
        assert c != 'clean_pair' and not c.startswith('next_') and c != TARGET
    return make_model('OLS', numeric=numeric, categorical=CATS).fit(frame[numeric + CATS], frame[TARGET])


def loro(numeric):
    rows = []
    for r in TRAIN_ROUNDS:
        tr, ho = train[train['round'] != r], train[train['round'] == r]
        mdl = fit(tr, numeric)
        rows.append({'held_out_round': r, 'n': len(ho),
                     'MAE_s': mean_absolute_error(ho[TARGET], mdl.predict(ho[numeric + CATS]))})
    return pd.DataFrame(rows)


VARIANTS = {'E0_reference': NUMERIC,
            'E1_gap_cap5': NUMERIC + ['gap_ahead_cap_s'],
            'E1b_gap_lt1': NUMERIC + ['gap_lt_1s']}
loro_tables = []
for name, numeric in VARIANTS.items():
    t = loro(numeric).assign(variant=name)
    loro_tables.append(t)
    trials.append({'stage': 'selection LORO train R1-7', 'variant': name, 'inputs': '+'.join(numeric[4:]) or '(6 ref)',
                   'model': 'OLS make_model', 'rows': f'train held-out rounds, n={t.n.sum()}',
                   'MAE_s': t.MAE_s.mean(), 'metric_note': 'mean of 7 held-out-race MAEs',
                   'used_validation': False, 'who': 'Valentín'})
loro_df = pd.concat(loro_tables)
loro_mean = loro_df.groupby('variant').MAE_s.mean()
chosen = 'E1b_gap_lt1' if loro_mean['E1b_gap_lt1'] < loro_mean['E1_gap_cap5'] - TIE_MARGIN_S else 'E1_gap_cap5'
cand_feat = {'E1_gap_cap5': 'gap_ahead_cap_s', 'E1b_gap_lt1': 'gap_lt_1s'}[chosen]
s1_feat = {'E1_gap_cap5': 'gap_ahead_nopit_cap_s', 'E1b_gap_lt1': 'gap_nopit_lt_1s'}[chosen]

# =============================================================== 4. validación (una sola vez)
train['placebo_feature'] = placebo(train, cand_feat)
val['placebo_feature'] = placebo(val, cand_feat)
check('K6 placebo: misma distribución por carrera, seed 414',
      all(np.allclose(np.sort(d.loc[d['round'] == r, 'placebo_feature']), np.sort(d.loc[d['round'] == r, cand_feat]))
          for d in (train, val) for r in d['round'].unique())
      and not np.allclose(val.placebo_feature, val[cand_feat]), 'permutación dentro de carrera')

RUNS = {
    'E0_reference_ols': ('OLS reference = matched control', NUMERIC),
    'E1_candidate': (f'Candidate: OLS + {cand_feat}', NUMERIC + [cand_feat]),
    'E2_placebo': (f'Placebo: OLS + {cand_feat} permuted within race', NUMERIC + ['placebo_feature']),
    'S1_sensitivity_nopit': (f'Sensitivity: OLS + {s1_feat} (ahead car not on pit in/out lap)', NUMERIC + [s1_feat]),
}
OUT.mkdir(exist_ok=True)
(OUT / 'models').mkdir(exist_ok=True)
models, preds = {}, pd.DataFrame({
    'pair_id': val.pair_id, 'round': val['round'], 'event_name': val.event_name,
    'driver_number': val.driver_number, 'current_lap_number': val.current_lap_number,
    'y_true_s': val[TARGET], 'current_lap_s': val.current_lap_s,
    'gap_ahead_s': val.gap_ahead_s, cand_feat: val[cand_feat], s1_feat: val[s1_feat],
    'placebo_feature': val.placebo_feature, 'persistence': val.current_lap_s})
for run, (desc, numeric) in RUNS.items():
    mdl = fit(train, numeric)
    models[run] = mdl
    preds[f'pred_{run}'] = mdl.predict(val[numeric + CATS])
    preds[f'abs_err_{run}'] = (preds.y_true_s - preds[f'pred_{run}']).abs()
    joblib.dump({'model': mdl, 'numeric': numeric, 'categorical': CATS, 'description': desc},
                OUT / 'models' / f'{run}.joblib')

scaler = models['E1_candidate'].named_steps['prepare'].named_transformers_['numeric'].named_steps['scale']
both = pd.concat([train, val])
check('K4 preprocesamiento ajustado solo con train (media del scaler para la feature)',
      np.isclose(scaler.mean_[-1], train[cand_feat].mean()) and not np.isclose(scaler.mean_[-1], both[cand_feat].mean()),
      f'scaler={scaler.mean_[-1]:.4f} train={train[cand_feat].mean():.4f} train+val={both[cand_feat].mean():.4f}')
check('K7 E0 reproduce el OLS de Hito 1', round(preds.abs_err_E0_reference_ols.mean(), 4) == HITO1_OLS_MAE,
      f'{preds.abs_err_E0_reference_ols.mean():.6f}')
check('K8 predicciones finitas, mismas 1.808 filas para todas las versiones',
      np.isfinite(preds.filter(like='pred_').to_numpy()).all() and preds.pair_id.equals(val.pair_id)
      and len(preds) == 1808, '1 predicción por versión y pair_id')

# ---------------------------------------------------------------- métricas
pops = [('all_validation', np.ones(len(preds), bool))] + [
    (f'R{r} {preds.loc[preds["round"] == r, "event_name"].iat[0]}', (preds['round'] == r).to_numpy()) for r in VAL_ROUNDS]
rows = []
for run in RUNS:
    for label, mask in pops:
        rows.append({'version': run, 'population': label, 'n': int(mask.sum()),
                     'MAE_s': preds.loc[mask, f'abs_err_{run}'].mean()})
metrics = pd.DataFrame(rows)

desc_inputs = {'E0_reference_ols': 'OLS · 6 published inputs (also the matched control: same OLS, same rows, without the traffic input)',
               'E1_candidate': f'OLS · 6 inputs + {cand_feat}',
               'E2_placebo': f'OLS · 6 inputs + {cand_feat} permuted within race (seed 414)',
               'S1_sensitivity_nopit': f'OLS · 6 inputs + {s1_feat}'}
role = {'E0_reference_ols': 'OLS reference + matched control (shared run)', 'E1_candidate': 'Candidate',
        'E2_placebo': 'Placebo control', 'S1_sensitivity_nopit': 'Sensitivity'}
comp = metrics[metrics.population == 'all_validation'].copy()
comp.insert(0, 'role', comp.version.map(role))
comp['model_and_inputs'] = comp.version.map(desc_inputs)
comp['output'] = 'salidas_hito2/validation_predictions.csv · col pred_' + comp.version
comp = comp[['role', 'version', 'model_and_inputs', 'n', 'MAE_s', 'output']]

# control − candidato, global y por carrera; MAE global reconstruido ponderando por n
race = []
for label, mask in pops:
    c, k = preds.loc[mask, 'abs_err_E0_reference_ols'].mean(), preds.loc[mask, 'abs_err_E1_candidate'].mean()
    pl = preds.loc[mask, 'abs_err_E2_placebo'].mean()
    race.append({'population': label, 'n': int(mask.sum()), 'control_MAE_s': c, 'candidate_MAE_s': k,
                 'control_minus_candidate_s': c - k, 'placebo_MAE_s': pl, 'control_minus_placebo_s': c - pl,
                 'share_rows_candidate_better': float((preds.loc[mask, 'abs_err_E1_candidate']
                                                       < preds.loc[mask, 'abs_err_E0_reference_ols']).mean())})
race = pd.DataFrame(race)
r8, r9 = race.iloc[1], race.iloc[2]
weighted = {k: (r8[k] * r8.n + r9[k] * r9.n) / (r8.n + r9.n) for k in ['control_MAE_s', 'candidate_MAE_s']}
check('K9 MAE global = promedio de R8/R9 ponderado por n',
      all(np.isclose(weighted[k], race.iloc[0][k]) for k in weighted), {k: round(v, 6) for k, v in weighted.items()})

# IC bootstrap 95 % de (control − candidato) por bloques piloto-carrera
rng = np.random.default_rng(RANDOM_SEED)
preds['block'] = preds['round'].astype(str) + '-' + preds.driver_number
blocks = preds.groupby('block')[['abs_err_E0_reference_ols', 'abs_err_E1_candidate']].agg(['sum', 'count'])
bc, bk, bn = (blocks[('abs_err_E0_reference_ols', 'sum')].to_numpy(),
              blocks[('abs_err_E1_candidate', 'sum')].to_numpy(), blocks[('abs_err_E0_reference_ols', 'count')].to_numpy())
boot = []
for _ in range(2000):
    i = rng.integers(0, len(bn), len(bn))
    boot.append((bc[i].sum() - bk[i].sum()) / bn[i].sum())
ci = {'control_minus_candidate_s': float(race.iloc[0].control_minus_candidate_s),
      'ci95_low_s': float(np.percentile(boot, 2.5)), 'ci95_high_s': float(np.percentile(boot, 97.5)),
      'blocks': int(len(bn)), 'resamples': 2000, 'seed': RANDOM_SEED, 'block_definition': 'round-driver'}

# MAE por tramo de gap (slice declarado en HITO1 §4)
bins = [-np.inf, 1, 2, GAP_CAP_S, np.inf]
preds['gap_bin'] = pd.cut(preds.gap_ahead_s, bins, right=False, labels=['<1s', '1-2s', '2-5s', '>=5s'])
gapbin = (preds.groupby('gap_bin', observed=True)
          .agg(n=('pair_id', 'size'), control_MAE_s=('abs_err_E0_reference_ols', 'mean'),
               candidate_MAE_s=('abs_err_E1_candidate', 'mean')).reset_index())
gapbin['control_minus_candidate_s'] = gapbin.control_MAE_s - gapbin.candidate_MAE_s

for run in RUNS:
    trials.append({'stage': 'validation R8-9 (single use)', 'variant': run, 'inputs': '+'.join(RUNS[run][1][4:]) or '(6 ref)',
                   'model': 'OLS make_model', 'rows': 'validation n=1808 (hash f453a98d…)',
                   'MAE_s': float(metrics.query('version == @run and population == "all_validation"').MAE_s.iat[0]),
                   'metric_note': 'pooled validation MAE', 'used_validation': True, 'who': 'Valentín'})

# coeficientes del candidato (mapeo de inputs codificados → originales, útil para SHAP)
coef = pd.DataFrame({'encoded_input': models['E1_candidate'].named_steps['prepare'].get_feature_names_out(),
                     'coefficient_standardized': models['E1_candidate'].named_steps['regression'].coef_})

# =============================================================== 5. guardar
preds.drop(columns='block').to_csv(OUT / 'validation_predictions.csv', index=False)
comp.to_csv(OUT / 'comparison_table.csv', index=False)
metrics.to_csv(OUT / 'metrics_by_version_and_race.csv', index=False)
race.to_csv(OUT / 'control_vs_candidate_by_race.csv', index=False)
gapbin.to_csv(OUT / 'mae_by_gap_bin.csv', index=False)
loro_df.to_csv(OUT / 'selection_loro_train.csv', index=False)
pd.DataFrame(trials).to_csv(OUT / 'trials_log.csv', index=False)
coverage.to_csv(OUT / 'feature_coverage.csv', index=False)
coef.to_csv(OUT / 'candidate_coefficients.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'checks.csv', index=False)
# features de train para fondo SHAP (Camilo, C5–C6): mismas columnas que entran al modelo
train[['pair_id', 'round'] + NUMERIC + ['gap_ahead_cap_s', 'gap_lt_1s', 'gap_ahead_nopit_cap_s', 'placebo_feature'] + CATS
      + [TARGET]].to_csv(OUT / 'train_features.csv', index=False)
val[['pair_id', 'round'] + NUMERIC + ['gap_ahead_cap_s', 'gap_lt_1s', 'gap_ahead_nopit_cap_s', 'placebo_feature'] + CATS
    + [TARGET]].to_csv(OUT / 'validation_features.csv', index=False)
(OUT / 'run.json').write_text(json.dumps({
    'script': 'hito2_experiment.py', 'random_seed': RANDOM_SEED,
    'train_rounds': [int(r) for r in train_rounds], 'validation_rounds': [int(r) for r in val_rounds],
    'n_train': len(train), 'n_validation': len(val),
    'n_validation_by_round': {int(k): int(v) for k, v in val['round'].value_counts().sort_index().items()},
    'validation_pair_id_set_sha256': set_hash(val.pair_id),
    'selection': {'rule': f'E1b only if its LORO mean MAE is lower than E1 by more than {TIE_MARGIN_S} s',
                  'loro_mean_MAE_s': loro_mean.round(6).to_dict(), 'chosen': chosen, 'candidate_feature': cand_feat},
    'bootstrap_control_minus_candidate': ci,
    'data_sha256': hashes, 'snapshots_manifest_events': [e.get('round') for e in manifest['events']],
    'test_data_used': False,
    'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                    'sklearn': sklearn.__version__, 'platform': platform.platform()},
    'checks_passed': len(checks)}, ensure_ascii=False, indent=2), encoding='utf-8')

pd.set_option('display.width', 200)
fmt = lambda x: f'{x:.4f}'  # noqa: E731
print('Selección LORO (train R1–7, MAE medio s):', loro_mean.round(4).to_dict(), '→', chosen)
print('\n', comp[['role', 'version', 'n', 'MAE_s']].to_string(index=False, float_format=fmt))
print('\n', race.to_string(index=False, float_format=fmt))
print('\nIC95 bootstrap control − candidato:', {k: round(v, 4) if isinstance(v, float) else v for k, v in ci.items()})
print('\n', gapbin.to_string(index=False, float_format=fmt))
print(f'\n{len(checks)} checks OK · validación usada una vez · sin test')
