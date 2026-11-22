"""Informe final · NextLap — check pedido en el feedback del Hito 2 (parte Valentín: C1, C2, C6).

Feedback (Hito 2): fijar ANTES de evaluar una representación del cambio del gap entre t−1 y t,
compararla con E0 sobre los mismos IDs con selección dentro de R1–R7, comprobar si reduce el error
en el tramo < 1 s y reportar el MAE total y por carrera. El pre-registro está en DECISION_LOG.md
(sección "Informe final", commit anterior a este script).

Qué hace, en orden:
  1. Verifica datos (SHA-256), split R1–7 / R8–9, conteos y unicidad de pair_id (igual que Hito 2).
  2. Reconstruye `gap_ahead_cap_s` exactamente como hito2_experiment.py (y lo comprueba contra
     salidas_hito2/*_features.csv) y construye `gap_delta_s = gap_cap(t) − gap_cap(t−1)` dentro de
     piloto/carrera/stint, sobre la línea de tiempo completa, antes del filtro clean_pair.
     Regla de faltantes pre-declarada: sin vuelta t−1 en el stint → 0.
  3. Selección LORO dentro de R1–R7 entre E3a (6 + delta) y E3b (6 + gap + delta).
  4. Validación R8–R9, una vez: E0 (referencia OLS), E1 (gap del Hito 2) y el candidato elegido,
     con su control pareado. MAE total, por carrera, por tramo de gap; IC bootstrap.
  5. Corrección prioritaria del placebo: 200 permutaciones del placebo de E1 (semillas 414–613)
     para ubicar la permutación única del Hito 2 dentro de esa distribución.

Reutiliza con atribución load_snapshots, build_pairs, make_model, NUMERIC, CATEGORICAL, TARGET de
base_NextLap/nextlap_reference_v1.py (helper del curso IIT414W).

Uso (desde la raíz del repo, después de hito2_experiment.py):  python final_experiment.py  → salidas_final/
"""
from pathlib import Path
import hashlib, json, platform, sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'base_NextLap'
H2 = ROOT / 'salidas_hito2'
OUT = ROOT / 'salidas_final'
sys.path.insert(0, str(BASE))
from nextlap_reference_v1 import (load_snapshots, build_pairs, make_model,  # noqa: E402
                                  NUMERIC, CATEGORICAL, TARGET)

RANDOM_SEED = 414
GAP_CAP_S = 5.0
TIE_MARGIN_S = 0.005
N_PLACEBO = 200
VAL_HASH = 'f453a98dc906904c'
HITO2 = {'E0': 0.3867069503640436, 'E1': 0.3870497048738373, 'E2_placebo': 0.3867847825431219}
TRAIN_ROUNDS, VAL_ROUNDS = list(range(1, 8)), [8, 9]
CATS = CATEGORICAL
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
check('F0a versión de datos (SHA-256)', hashes == EXPECTED_SHA, {k: v[:12] for k, v in hashes.items()})
train = pd.read_csv(BASE / 'train_clean_v1.csv', dtype={'driver_number': str})
val = pd.read_csv(BASE / 'validation_clean_v1.csv', dtype={'driver_number': str})
tr_r, va_r = [int(r) for r in sorted(train['round'].unique())], [int(r) for r in sorted(val['round'].unique())]
check('F0b split: train R1–7, validación R8–9', tr_r == TRAIN_ROUNDS and va_r == VAL_ROUNDS, f'train={tr_r} val={va_r}')
check('F0c conteos', (len(train), len(val), int((val['round'] == 8).sum()), int((val['round'] == 9).sum()))
      == (5772, 1808, 1066, 742), f'train={len(train)} val={len(val)}')
check('F0d pair_id único y disjunto', train.pair_id.is_unique and val.pair_id.is_unique
      and not set(train.pair_id) & set(val.pair_id), 'train ∩ val = ∅')

# =============================================================== 2. features en la línea de tiempo original
raw, manifest = load_snapshots(BASE)
raw['round'] = raw['round'].astype(int)
laps = (raw.dropna(subset=['end_time_s']).sort_values(['round', 'end_time_s', 'driver_number'])
        .reset_index(drop=True).copy())
laps['ahead_end_time_s'] = laps.groupby('round').end_time_s.shift(1)
laps['gap_ahead_s'] = laps.end_time_s - laps.ahead_end_time_s
laps['gap_cap'] = laps.gap_ahead_s.fillna(GAP_CAP_S).clip(upper=GAP_CAP_S)   # regla Hito 2
# Prioridad de la Demo: auto de adelante en la MISMA vuelta (ignora autos doblados o que nos doblaron)
laps['ahead_lap_number'] = laps.groupby('round').lap_number.shift(1)
laps['ahead_other_lap'] = laps.ahead_lap_number.notna() & (laps.ahead_lap_number != laps.lap_number)
laps['ahead_samelap_end_time_s'] = laps.groupby(['round', 'lap_number']).end_time_s.shift(1)
laps['ahead_samelap_driver'] = laps.groupby(['round', 'lap_number']).driver_number.shift(1)
laps['gap_samelap_s'] = laps.end_time_s - laps.ahead_samelap_end_time_s
laps['samelap_missing'] = laps.gap_samelap_s.isna()
laps['gap_samelap_cap_s'] = laps.gap_samelap_s.fillna(GAP_CAP_S).clip(upper=GAP_CAP_S)  # regla pre-declarada

# t−1 del mismo piloto, misma carrera y mismo stint (lap_number = t − 1)
prev = laps.loc[laps.stint.notna(), ['round', 'driver_number', 'stint', 'lap_number', 'gap_cap', 'end_time_s',
                                       'ahead_end_time_s']].copy()
prev['lap_number'] = prev.lap_number + 1          # así se une con la vuelta t
prev = prev.rename(columns={'gap_cap': 'gap_cap_prev', 'end_time_s': 'end_time_prev_s',
                            'ahead_end_time_s': 'ahead_end_time_prev_s'})
laps = laps.merge(prev, on=['round', 'driver_number', 'stint', 'lap_number'], how='left', validate='one_to_one')
laps['delta_missing'] = laps.gap_cap_prev.isna()
laps['gap_delta_s'] = (laps.gap_cap - laps.gap_cap_prev).fillna(0.0)          # regla pre-declarada

m = ~laps.delta_missing
check('F1 temporal: t−1 y su auto de adelante terminaron antes del fin de t',
      (laps.end_time_prev_s[m] < laps.end_time_s[m]).all()
      and (laps.ahead_end_time_prev_s[m].dropna() <= laps.end_time_s[m][laps.ahead_end_time_prev_s[m].notna()]).all()
      and (laps.ahead_end_time_s.dropna() <= laps.end_time_s[laps.ahead_end_time_s.notna()]).all(),
      f'máx(end_time_t−1 − end_time_t) = {(laps.end_time_prev_s - laps.end_time_s).max():.3f} s (< 0)')
ms = laps.ahead_samelap_end_time_s.notna()
check('F1b misma vuelta: el auto usado completó la misma vuelta, antes del fin de t, y es otro piloto',
      (laps.ahead_samelap_end_time_s[ms] <= laps.end_time_s[ms]).all()
      and (laps.ahead_samelap_driver[ms] != laps.driver_number[ms]).all(), f'{int(ms.sum())} vueltas con auto en la misma vuelta')

pairs = build_pairs(raw)
pairs['round'] = pairs['round'].astype(int)
pairs = pairs.merge(laps[['round', 'driver_number', 'lap_number', 'gap_ahead_s', 'gap_cap', 'gap_cap_prev',
                          'gap_delta_s', 'delta_missing', 'ahead_other_lap', 'gap_samelap_s',
                          'samelap_missing', 'gap_samelap_cap_s']].rename(columns={'lap_number': 'current_lap_number'}),
                    on=['round', 'driver_number', 'current_lap_number'], how='left', validate='one_to_one')
cols = ['pair_id', 'gap_ahead_s', 'gap_cap', 'gap_cap_prev', 'gap_delta_s', 'delta_missing',
        'ahead_other_lap', 'gap_samelap_s', 'samelap_missing', 'gap_samelap_cap_s']
n0 = (len(train), len(val))
train = train.merge(pairs[cols], on='pair_id', how='left', validate='one_to_one')
val = val.merge(pairs[cols], on='pair_id', how='left', validate='one_to_one')
check('F2a join por pair_id sin filas perdidas ni duplicadas',
      (len(train), len(val)) == n0 and train.pair_id.is_unique and val.pair_id.is_unique,
      f'train {n0[0]}→{len(train)}, val {n0[1]}→{len(val)}')
check('F2b mismo set de validación que Hito 1/Hito 2 (hash)', set_hash(val.pair_id).startswith(VAL_HASH),
      set_hash(val.pair_id)[:16])
train['gap_ahead_cap_s'], val['gap_ahead_cap_s'] = train.gap_cap, val.gap_cap
h2tr, h2va = pd.read_csv(H2 / 'train_features.csv'), pd.read_csv(H2 / 'validation_features.csv')
check('F2c gap_ahead_cap_s idéntico al del Hito 2',
      np.allclose(train.set_index('pair_id').loc[h2tr.pair_id, 'gap_ahead_cap_s'], h2tr.gap_ahead_cap_s)
      and np.allclose(val.set_index('pair_id').loc[h2va.pair_id, 'gap_ahead_cap_s'], h2va.gap_ahead_cap_s),
      'misma feature que E1')
check('F3 sin fuga: ningún input next_* / clean_pair; delta sin clean_pair', True,
      'make_model bloquea next_*; gap_delta_s se calcula en `laps` (todas las vueltas) antes de unir a los pares clean')
coverage = pd.DataFrame([{'population': name, 'n': len(d), 'delta_fallback_rows': int(d.delta_missing.sum()),
                          'delta_fallback_share': float(d.delta_missing.mean()),
                          'delta_mean_s': float(d.gap_delta_s.mean()), 'delta_sd_s': float(d.gap_delta_s.std()),
                          'samelap_fallback_rows': int(d.samelap_missing.sum()),
                          'share_hito2_ahead_car_other_lap': float(d.ahead_other_lap.mean()),
                          'share_samelap_gap_lt_1s': float((d.gap_samelap_s < 1).mean())}
                         for name, d in [('train R1-7', train), ('validation R8-9', val)]])
other_lap_by_round = (pd.concat([train, val]).groupby(['round', 'event_name'])
                      .agg(n=('pair_id', 'size'), share_hito2_ahead_car_other_lap=('ahead_other_lap', 'mean'),
                           share_gap_lt_1s_hito2=('gap_ahead_s', lambda g: (g < 1).mean()),
                           share_gap_lt_1s_samelap=('gap_samelap_s', lambda g: (g < 1).mean())).reset_index())
check('F4 faltantes contados (regla: sin t−1 en el stint → 0)', True,
      coverage[['population', 'delta_fallback_rows']].to_dict('records'))


def fit(frame, numeric):
    for c in numeric + CATS:
        assert c != 'clean_pair' and not c.startswith('next_') and c != TARGET
    return make_model('OLS', numeric=numeric, categorical=CATS).fit(frame[numeric + CATS], frame[TARGET])


# =============================================================== 3. selección LORO dentro de R1–R7
VARIANTS = {'E0_reference': NUMERIC,
            'E1_gap_cap5': NUMERIC + ['gap_ahead_cap_s'],
            'E3a_delta': NUMERIC + ['gap_delta_s'],
            'E3b_gap_plus_delta': NUMERIC + ['gap_ahead_cap_s', 'gap_delta_s'],
            'E4_samelap_gap': NUMERIC + ['gap_samelap_cap_s']}
loro = []
for name, numeric in VARIANTS.items():
    for r in TRAIN_ROUNDS:
        tr, ho = train[train['round'] != r], train[train['round'] == r]
        loro.append({'variant': name, 'held_out_round': r, 'n': len(ho),
                     'MAE_s': mean_absolute_error(ho[TARGET], fit(tr, numeric).predict(ho[numeric + CATS]))})
loro = pd.DataFrame(loro)
loro_mean = loro.groupby('variant', sort=False).MAE_s.mean()
for name, numeric in VARIANTS.items():
    trials.append({'stage': 'final · selection LORO train R1-7', 'variant': name,
                   'inputs': '+'.join(numeric[4:]) or '(6 ref)', 'model': 'OLS make_model',
                   'rows': 'train held-out rounds, n=5772', 'MAE_s': loro_mean[name],
                   'metric_note': 'mean of 7 held-out-race MAEs', 'used_validation': False, 'who': 'Valentín'})
chosen = 'E3b_gap_plus_delta' if loro_mean['E3b_gap_plus_delta'] < loro_mean['E3a_delta'] - TIE_MARGIN_S else 'E3a_delta'
control = {'E3a_delta': 'E0_reference', 'E3b_gap_plus_delta': 'E1_gap_cap5'}[chosen]

# =============================================================== 4. validación (una vez)
RUNS = ['E0_reference', 'E1_gap_cap5', chosen, 'E4_samelap_gap']
preds = pd.DataFrame({'pair_id': val.pair_id, 'round': val['round'], 'event_name': val.event_name,
                      'driver_number': val.driver_number, 'current_lap_number': val.current_lap_number,
                      'y_true_s': val[TARGET], 'gap_ahead_s': val.gap_ahead_s,
                      'gap_ahead_cap_s': val.gap_ahead_cap_s, 'gap_cap_prev': val.gap_cap_prev,
                      'gap_delta_s': val.gap_delta_s, 'delta_fallback': val.delta_missing,
                      'ahead_other_lap': val.ahead_other_lap, 'gap_samelap_s': val.gap_samelap_s,
                      'gap_samelap_cap_s': val.gap_samelap_cap_s, 'samelap_fallback': val.samelap_missing})
models = {}
for run in RUNS:
    models[run] = fit(train, VARIANTS[run])
    preds[f'pred_{run}'] = models[run].predict(val[VARIANTS[run] + CATS])
    preds[f'abs_err_{run}'] = (preds.y_true_s - preds[f'pred_{run}']).abs()
check('F5 E0 y E1 reproducen el MAE del Hito 2',
      np.isclose(preds.abs_err_E0_reference.mean(), HITO2['E0'], atol=1e-9)
      and np.isclose(preds.abs_err_E1_gap_cap5.mean(), HITO2['E1'], atol=1e-9),
      f'E0={preds.abs_err_E0_reference.mean():.6f} E1={preds.abs_err_E1_gap_cap5.mean():.6f}')
h2p = pd.read_csv(H2 / 'validation_predictions.csv')
check('F6 mismas filas, mismo orden y mismas predicciones E0 que validation_predictions.csv del Hito 2',
      preds.pair_id.equals(h2p.pair_id) and np.allclose(preds.pred_E0_reference, h2p.pred_E0_reference_ols, atol=1e-9),
      '1.808 filas')
scaler = models[chosen].named_steps['prepare'].named_transformers_['numeric'].named_steps['scale']
both = pd.concat([train, val])
check('F7 preprocesamiento ajustado solo con train (media del scaler para gap_delta_s)',
      np.isclose(scaler.mean_[-1], train.gap_delta_s.mean()) and not np.isclose(scaler.mean_[-1], both.gap_delta_s.mean()),
      f'scaler={scaler.mean_[-1]:.5f} train={train.gap_delta_s.mean():.5f} train+val={both.gap_delta_s.mean():.5f}')
sc4 = models['E4_samelap_gap'].named_steps['prepare'].named_transformers_['numeric'].named_steps['scale']
check('F7b preprocesamiento de E4 ajustado solo con train',
      np.isclose(sc4.mean_[-1], train.gap_samelap_cap_s.mean()) and not np.isclose(sc4.mean_[-1], both.gap_samelap_cap_s.mean()),
      f'scaler={sc4.mean_[-1]:.5f} train={train.gap_samelap_cap_s.mean():.5f} train+val={both.gap_samelap_cap_s.mean():.5f}')
check('F8 predicciones finitas', np.isfinite(preds.filter(like='pred_').to_numpy()).all(), '')

pops = [('all_validation', np.ones(len(preds), bool))] + [
    (f'R{r} {preds.loc[preds["round"] == r, "event_name"].iat[0]}', (preds['round'] == r).to_numpy()) for r in VAL_ROUNDS]


def contrast(ctrl, cand):
    rows = []
    for label, mask in pops:
        c, k = preds.loc[mask, f'abs_err_{ctrl}'].mean(), preds.loc[mask, f'abs_err_{cand}'].mean()
        rows.append({'control': ctrl, 'candidate': cand, 'population': label, 'n': int(mask.sum()),
                     'control_MAE_s': c, 'candidate_MAE_s': k, 'control_minus_candidate_s': c - k,
                     'share_rows_candidate_better': float((preds.loc[mask, f'abs_err_{cand}']
                                                           < preds.loc[mask, f'abs_err_{ctrl}']).mean())})
    out = pd.DataFrame(rows)
    r8, r9 = out.iloc[1], out.iloc[2]
    for k in ['control_MAE_s', 'candidate_MAE_s']:
        check(f'F9 {cand} vs {ctrl}: {k} global = R8/R9 ponderado por n',
              np.isclose((r8[k] * r8.n + r9[k] * r9.n) / (r8.n + r9.n), out.iloc[0][k]), '')
    return out


def boot_ci(ctrl, cand):
    rng = np.random.default_rng(RANDOM_SEED)
    blk = preds.assign(b=preds['round'].astype(str) + '-' + preds.driver_number).groupby('b')
    a, b, n = (blk[f'abs_err_{ctrl}'].sum().to_numpy(), blk[f'abs_err_{cand}'].sum().to_numpy(), blk.size().to_numpy())
    bs = [(a[i].sum() - b[i].sum()) / n[i].sum() for i in (rng.integers(0, len(n), len(n)) for _ in range(2000))]
    return {'control': ctrl, 'candidate': cand,
            'control_minus_candidate_s': float(preds[f'abs_err_{ctrl}'].mean() - preds[f'abs_err_{cand}'].mean()),
            'ci95_low_s': float(np.percentile(bs, 2.5)), 'ci95_high_s': float(np.percentile(bs, 97.5)),
            'blocks': int(len(n)), 'resamples': 2000, 'seed': RANDOM_SEED}


race = pd.concat([contrast(control, chosen)] + ([contrast('E0_reference', chosen)] if control != 'E0_reference' else [])
                 + [contrast('E0_reference', 'E4_samelap_gap'), contrast('E1_gap_cap5', 'E4_samelap_gap')])
cis = ([boot_ci(control, chosen)] + ([boot_ci('E0_reference', chosen)] if control != 'E0_reference' else [])
       + [boot_ci('E0_reference', 'E4_samelap_gap'), boot_ci('E1_gap_cap5', 'E4_samelap_gap')])

bins = [-np.inf, 1, 2, GAP_CAP_S, np.inf]
preds['gap_bin'] = pd.cut(preds.gap_ahead_s, bins, right=False, labels=['<1s', '1-2s', '2-5s', '>=5s']).astype(str)
gb = preds.groupby('gap_bin', sort=False)
gapbin = pd.DataFrame({'n': gb.size(), **{f'MAE_{r}_s': gb[f'abs_err_{r}'].mean() for r in RUNS}}).reindex(
    ['<1s', '1-2s', '2-5s', '>=5s']).reset_index()
gapbin['control_minus_candidate_s'] = gapbin[f'MAE_{control}_s'] - gapbin[f'MAE_{chosen}_s']
gapbin['E0_minus_candidate_s'] = gapbin['MAE_E0_reference_s'] - gapbin[f'MAE_{chosen}_s']
gapbin['E0_minus_E4_s'] = gapbin['MAE_E0_reference_s'] - gapbin['MAE_E4_samelap_gap_s']
# mismos tramos pero definidos con el gap en la misma vuelta (la definición de E4)
preds['gap_bin_samelap'] = pd.cut(preds.gap_samelap_s.fillna(GAP_CAP_S), bins, right=False,
                                  labels=['<1s', '1-2s', '2-5s', '>=5s']).astype(str)
g4 = preds.groupby('gap_bin_samelap', sort=False)
gapbin_samelap = pd.DataFrame({'n': g4.size(), 'MAE_E0_reference_s': g4.abs_err_E0_reference.mean(),
                               'MAE_E4_samelap_gap_s': g4.abs_err_E4_samelap_gap.mean()}).reindex(
    ['<1s', '1-2s', '2-5s', '>=5s']).reset_index()
gapbin_samelap['E0_minus_E4_s'] = gapbin_samelap.MAE_E0_reference_s - gapbin_samelap.MAE_E4_samelap_gap_s

for run in RUNS:
    trials.append({'stage': 'final · validation R8-9 (single use)', 'variant': run,
                   'inputs': '+'.join(VARIANTS[run][4:]) or '(6 ref)', 'model': 'OLS make_model',
                   'rows': 'validation n=1808 (hash f453a98d…)', 'MAE_s': preds[f'abs_err_{run}'].mean(),
                   'metric_note': 'pooled validation MAE', 'used_validation': True, 'who': 'Valentín'})

# =============================================================== 5. placebo de E1 con 200 permutaciones
def permute(d, col, seed):
    rng = np.random.default_rng(seed)
    out = d[col].copy()
    for r in sorted(d['round'].unique()):
        idx = d.index[d['round'] == r]
        out.loc[idx] = rng.permutation(d.loc[idx, col].to_numpy())
    return out


plac = []
for seed in range(RANDOM_SEED, RANDOM_SEED + N_PLACEBO):
    tr, va = train.copy(), val.copy()
    tr['placebo_feature'], va['placebo_feature'] = permute(tr, 'gap_ahead_cap_s', seed), permute(va, 'gap_ahead_cap_s', seed)
    p = fit(tr, NUMERIC + ['placebo_feature']).predict(va[NUMERIC + CATS + ['placebo_feature']])
    plac.append({'seed': seed, 'MAE_s': float(np.mean(np.abs(va[TARGET] - p)))})
plac = pd.DataFrame(plac)
check('F10 la semilla 414 reproduce el placebo único del Hito 2', np.isclose(plac.MAE_s.iat[0], HITO2['E2_placebo'], atol=1e-9),
      f'{plac.MAE_s.iat[0]:.7f}')
plac_summary = {
    'permutations': N_PLACEBO, 'seeds': f'{RANDOM_SEED}–{RANDOM_SEED + N_PLACEBO - 1}',
    'E0_MAE_s': HITO2['E0'], 'E1_MAE_s': HITO2['E1'], 'hito2_single_placebo_MAE_s': HITO2['E2_placebo'],
    'placebo_MAE_mean_s': float(plac.MAE_s.mean()), 'placebo_MAE_p2_5_s': float(plac.MAE_s.quantile(.025)),
    'placebo_MAE_p97_5_s': float(plac.MAE_s.quantile(.975)), 'placebo_MAE_min_s': float(plac.MAE_s.min()),
    'placebo_MAE_max_s': float(plac.MAE_s.max()),
    'share_placebo_better_than_E0': float((plac.MAE_s < HITO2['E0']).mean()),
    'share_placebo_better_than_E1': float((plac.MAE_s < HITO2['E1']).mean()),
    'percentile_of_hito2_permutation': float((plac.MAE_s <= HITO2['E2_placebo']).mean()),
}

# =============================================================== 6. guardar
OUT.mkdir(exist_ok=True)
preds.to_csv(OUT / 'validation_predictions_final.csv', index=False)
loro.to_csv(OUT / 'selection_loro_train_final.csv', index=False)
race.to_csv(OUT / 'control_vs_candidate_by_race_final.csv', index=False)
gapbin.to_csv(OUT / 'mae_by_gap_bin_final.csv', index=False)
coverage.to_csv(OUT / 'feature_coverage_final.csv', index=False)
other_lap_by_round.to_csv(OUT / 'ahead_car_other_lap_by_round.csv', index=False)
gapbin_samelap.to_csv(OUT / 'mae_by_samelap_gap_bin_E4.csv', index=False)
plac.to_csv(OUT / 'placebo_E1_permutations.csv', index=False)
pd.DataFrame(trials).to_csv(OUT / 'trials_log_final.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'checks_final.csv', index=False)
pd.DataFrame({'encoded_input': models[chosen].named_steps['prepare'].get_feature_names_out(),
              'coefficient_standardized': models[chosen].named_steps['regression'].coef_}).to_csv(
    OUT / 'candidate_final_coefficients.csv', index=False)
(OUT / 'run_final.json').write_text(json.dumps({
    'script': 'final_experiment.py', 'random_seed': RANDOM_SEED,
    'train_rounds': tr_r, 'validation_rounds': va_r, 'n_train': len(train), 'n_validation': len(val),
    'validation_pair_id_set_sha256': set_hash(val.pair_id),
    'feature': 'gap_delta_s = gap_cap(t) − gap_cap(t−1), mismo piloto/carrera/stint; sin t−1 → 0',
    'selection': {'rule': f'E3b solo si su MAE LORO medio < E3a − {TIE_MARGIN_S} s',
                  'loro_mean_MAE_s': loro_mean.round(6).to_dict(), 'chosen': chosen, 'matched_control': control},
    'bootstrap': cis, 'placebo_E1': plac_summary, 'data_sha256': hashes, 'test_data_used': False,
    'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                    'sklearn': sklearn.__version__, 'platform': platform.platform()},
    'checks_passed': len(checks)}, ensure_ascii=False, indent=2), encoding='utf-8')

pd.set_option('display.width', 200)
f4 = lambda x: f'{x:.5f}'  # noqa: E731
print('Cobertura delta:\n', coverage.to_string(index=False, float_format=f4))
print('\nLORO (train R1–7):', loro_mean.round(5).to_dict(), '→', chosen, '| control:', control)
print('\n', race.to_string(index=False, float_format=f4))
print('\nIC95:', cis)
print('\n', gapbin.to_string(index=False, float_format=f4))
print('\n', other_lap_by_round.to_string(index=False, float_format=f4))
print('\n', gapbin_samelap.to_string(index=False, float_format=f4))
print('\nPlacebo E1:', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in plac_summary.items()})
print(f'\n{len(checks)} checks OK · validación usada una vez · sin test')
