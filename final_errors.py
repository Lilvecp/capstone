"""Informe final · NextLap — errores por contexto (parte Camilo: FINAL_REPORT §3).

Lee las predicciones ya generadas por final_experiment.py (no reajusta modelos ni usa el target para
elegir nada) y produce las tablas de §3:

  1. Contraste por carrera E0 vs E1 y E0 vs E4 (n, MAE, control − candidato) y comprobación de que el
     global es el promedio de R8/R9 ponderado por n.
  2. Error por tipo de "auto de adelante": misma vuelta vs otra vuelta, y dentro de otra vuelta,
     si el piloto está doblando a un auto más atrasado o lo está doblando uno más adelantado.
     La dirección se obtiene del `lap_number` del auto que cruzó justo antes (misma línea de tiempo
     completa que hito2_experiment.py, antes del filtro clean_pair).
  3. Dos casos inspeccionados con |error| recalculado a mano: `2026-R8-3-L10` (auto de adelante en la
     misma vuelta) y `2026-R8-81-L69` (auto de adelante en otra vuelta, gap < 1 s).

Es un análisis descriptivo sobre validación R8–R9: no se usa para elegir modelo (la decisión ya está
tomada en §2 con los criterios pre-registrados).

Uso (desde la raíz, después de final_experiment.py):  python final_errors.py  → salidas_final/errors_*
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'base_NextLap'
OUT = ROOT / 'salidas_final'
sys.path.insert(0, str(BASE))
from nextlap_reference_v1 import load_snapshots  # noqa: E402  helper del curso IIT414W

CASES = ['2026-R8-3-L10', '2026-R8-81-L69']
MODELS = {'E0_reference': 'E0', 'E1_gap_cap5': 'E1', 'E4_samelap_gap': 'E4'}
checks = []


def check(name, ok, detail=''):
    checks.append({'check': name, 'passed': bool(ok), 'detail': str(detail)})
    print(('OK  ' if ok else 'FAIL'), name, detail)


pred = pd.read_csv(OUT / 'validation_predictions_final.csv')
check('C0 filas de validación', len(pred) == 1808 and pred.pair_id.is_unique, f'n={len(pred)}')

# --- 1. Contraste por carrera ------------------------------------------------------------------
rows = []
for ctrl, cand in [('E0_reference', 'E1_gap_cap5'), ('E0_reference', 'E4_samelap_gap')]:
    for pop, d in [('all_validation', pred)] + [(f'R{r} {n}', g) for (r, n), g in pred.groupby(['round', 'event_name'])]:
        e_c, e_k = d[f'abs_err_{ctrl}'], d[f'abs_err_{cand}']
        rows.append({'control': ctrl, 'candidate': cand, 'population': pop, 'n': len(d),
                     'control_MAE_s': e_c.mean(), 'candidate_MAE_s': e_k.mean(),
                     'control_minus_candidate_s': e_c.mean() - e_k.mean(),
                     'share_rows_candidate_better': float((e_k < e_c).mean())})
by_race = pd.DataFrame(rows)
for cand, g in by_race.groupby('candidate'):
    races = g[g.population != 'all_validation']
    weighted = (races.n * races.control_minus_candidate_s).sum() / races.n.sum()
    glob = g[g.population == 'all_validation'].control_minus_candidate_s.iloc[0]
    check(f'C1 global = R8/R9 ponderado ({cand})', np.isclose(weighted, glob, atol=1e-12),
          f'{glob * 1000:+.3f} ms')
ref = pd.read_csv(OUT / 'control_vs_candidate_by_race_final.csv')
r4 = ref[(ref.control == 'E0_reference') & (ref.candidate == 'E4_samelap_gap')].set_index('population')
m4 = by_race[by_race.candidate == 'E4_samelap_gap'].set_index('population')
check('C2 coincide con control_vs_candidate_by_race_final.csv (E4)',
      np.allclose(r4.control_minus_candidate_s, m4.loc[r4.index].control_minus_candidate_s, atol=1e-12))

# --- 2. Tipo de auto de adelante ----------------------------------------------------------------
raw, _ = load_snapshots(BASE)          # misma línea de tiempo y orden que final_experiment.py
raw['round'] = raw['round'].astype(int)
laps = (raw.dropna(subset=['end_time_s']).sort_values(['round', 'end_time_s', 'driver_number'])
        .reset_index(drop=True).copy())
laps['ahead_lap_number'] = laps.groupby('round').lap_number.shift(1)
laps = laps[laps['round'].isin([8, 9])].dropna(subset=['lap_number'])
laps['pair_id'] = [f'{s}-R{r}-{n}-L{int(l)}' for s, r, n, l in
                   zip(laps.season, laps['round'], laps.driver_number, laps.lap_number)]
pred = pred.merge(laps[['pair_id', 'ahead_lap_number']], on='pair_id', how='left', validate='one_to_one')
same = pred.ahead_lap_number == pred.current_lap_number
check('C3 dirección reconstruida coincide con ahead_other_lap', (~same == pred.ahead_other_lap).all(),
      f'{int((~same).sum())} filas en otra vuelta')
pred['ahead_type'] = np.select(
    [same, pred.ahead_lap_number < pred.current_lap_number, pred.ahead_lap_number > pred.current_lap_number],
    ['misma vuelta', 'otra vuelta: dobla a un atrasado', 'otra vuelta: lo dobla uno adelantado'],
    'sin auto antes')
pred['gap_lt_2s'] = pred.gap_ahead_s < 2

ctx = []
for (sub, d) in [('todas', pred), ('gap < 2 s', pred[pred.gap_lt_2s]), ('gap < 1 s', pred[pred.gap_ahead_s < 1])]:
    for t, g in d.groupby('ahead_type'):
        r = {'subset': sub, 'ahead_type': t, 'n': len(g)}
        for col, short in MODELS.items():
            r[f'MAE_{short}_s'] = g[f'abs_err_{col}'].mean()
        r['E0_minus_E1_s'] = r['MAE_E0_s'] - r['MAE_E1_s']
        r['E0_minus_E4_s'] = r['MAE_E0_s'] - r['MAE_E4_s']
        r['mean_shift_E1_minus_E0_s'] = (g.pred_E1_gap_cap5 - g.pred_E0_reference).mean()
        r['share_obs_slower_than_E0'] = float((g.y_true_s > g.pred_E0_reference).mean())
        ctx.append(r)
ctx = pd.DataFrame(ctx)
check('C4 subconjuntos suman n', ctx[ctx.subset == 'todas'].n.sum() == 1808)

# --- 3. Casos inspeccionados --------------------------------------------------------------------
case_rows = []
for pid in CASES:
    r = pred.set_index('pair_id').loc[pid]
    row = {'pair_id': pid, 'event_name': r.event_name, 'driver_number': int(r.driver_number),
           'current_lap_number': int(r.current_lap_number), 'ahead_type': r.ahead_type,
           'gap_ahead_s': r.gap_ahead_s, 'gap_samelap_s': r.gap_samelap_s,
           'gap_samelap_cap_s': r.gap_samelap_cap_s, 'y_true_s': r.y_true_s}
    for col, short in MODELS.items():
        row[f'pred_{short}'] = r[f'pred_{col}']
        row[f'abs_err_{short}_hand'] = abs(r.y_true_s - r[f'pred_{col}'])
        check(f'C5 |error| a mano = archivo ({pid}, {short})',
              np.isclose(row[f'abs_err_{short}_hand'], r[f'abs_err_{col}'], atol=1e-9),
              f'{row[f"abs_err_{short}_hand"]:.3f} s')
    case_rows.append(row)
cases = pd.DataFrame(case_rows)

OUT.mkdir(exist_ok=True)
by_race.to_csv(OUT / 'errors_by_race_E1_E4.csv', index=False)
ctx.to_csv(OUT / 'errors_by_ahead_type.csv', index=False)
cases.to_csv(OUT / 'errors_cases.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'errors_checks.csv', index=False)
print(ctx.round(5).to_string(index=False))
print(cases.T.to_string())
n_ok = sum(c['passed'] for c in checks)
print(f'\n{n_ok}/{len(checks)} checks OK · descriptivo, sin reajustar modelos')
sys.exit(0 if n_ok == len(checks) else 1)
