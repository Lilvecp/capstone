"""Hito 1 · NextLap — chequeo de factibilidad de la hipótesis de tráfico (NO es la extensión).

Qué hace:
  1. Reconstruye los pares desde los snapshots originales (`base_NextLap/data/`) con el helper
     del curso (`load_snapshots`, `build_pairs`), es decir, sobre la línea de tiempo original y
     ANTES del filtro retrospectivo `clean_pair`, y verifica que el set de `pair_id` clean coincide
     exactamente con los CSV publicados de train y validación.
  2. Calcula la feature candidata `gap_ahead_s`: segundos entre que el auto inmediatamente
     anterior (otro piloto, misma carrera) cruzó la línea y que este piloto terminó la vuelta t.
     Se usa SOLO `end_time_s` de otros autos con `end_time_s <= end_time_s(t)` → disponible en t.
  3. Reporta cobertura (train y validación) y distribución (solo train). No ajusta modelos,
     no mira `next_lap_s` de validación y no calcula ningún MAE de la extensión (eso es Hito 2).

Uso (desde la raíz del repo):  python hito1_feasibility_traffic.py   → salidas_hito1/feasibility_traffic_*.
"""
from pathlib import Path
import hashlib, json, sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'base_NextLap'
OUT = ROOT / 'salidas_hito1'
sys.path.insert(0, str(BASE))
from nextlap_reference_v1 import load_snapshots, build_pairs  # noqa: E402

GAP_CAP_S = 5.0  # representación pre-declarada en HITO1.md §3 (fijada antes de ver resultados)

checks = []


def check(name, ok, detail=''):
    checks.append({'check': name, 'passed': bool(ok), 'detail': str(detail)})
    if not ok:
        raise AssertionError(f'{name}: {detail}')


raw, manifest = load_snapshots(BASE)  # verifica SHA-256 de cada snapshot contra el manifest
raw['round'] = raw['round'].astype(int)
check('F1 snapshots R1–R9 sin test', set(raw['round']) == set(range(1, 10)),
      f"rondas={sorted(set(raw['round']))}")

# ---------------------------------------------------------------- feature sobre la línea de tiempo completa
# Todas las vueltas (incluidas pit, SC, no precisas): el tráfico físico existe aunque la vuelta no sea clean.
laps = raw.dropna(subset=['end_time_s']).sort_values(['round', 'end_time_s', 'driver_number']).copy()
laps['prev_end_time_s'] = laps.groupby('round').end_time_s.shift(1)
laps['prev_driver'] = laps.groupby('round').driver_number.shift(1)
laps['gap_ahead_s'] = laps.end_time_s - laps.prev_end_time_s
check('F2 timestamp usado <= fin de t', (laps.prev_end_time_s.dropna() <= laps.end_time_s[laps.prev_end_time_s.notna()]).all(),
      'el auto de adelante siempre cruzó antes o a la vez')
check('F3 nunca el mismo piloto', (laps.prev_driver.dropna() != laps.driver_number[laps.prev_driver.notna()]).all(),
      'el auto anterior es otro piloto')

# ---------------------------------------------------------------- pares (mismo helper que la referencia)
pairs = build_pairs(raw)
pairs['round'] = pairs['round'].astype(int)
key = ['round', 'driver_number', 'lap_number']
pairs = pairs.merge(laps[key + ['gap_ahead_s']].rename(columns={'lap_number': 'current_lap_number'}),
                    on=['round', 'driver_number', 'current_lap_number'], how='left', validate='one_to_one')
clean = pairs[pairs.clean_pair]

train_rel = pd.read_csv(BASE / 'train_clean_v1.csv', dtype={'driver_number': str})
val_rel = pd.read_csv(BASE / 'validation_clean_v1.csv', dtype={'driver_number': str})
tr = clean[clean['round'].le(7)]
va = clean[clean['round'].isin([8, 9])]
check('F4 mismos pair_id que el release (train)', set(tr.pair_id) == set(train_rel.pair_id), f'n={len(tr)}')
check('F5 mismos pair_id que el release (validación)', set(va.pair_id) == set(val_rel.pair_id), f'n={len(va)}')
val_hash = hashlib.sha256('\n'.join(sorted(va.pair_id)).encode()).hexdigest()
check('F6 hash del set de validación = referencia', val_hash.startswith('f453a98dc906904c'), val_hash[:16])
m = tr.merge(train_rel[['pair_id', 'recent_mean_s']], on='pair_id', suffixes=('', '_rel'))
check('F7 recent_mean_s reconstruido = publicado', np.allclose(m.recent_mean_s, m.recent_mean_s_rel),
      f'max |dif| = {(m.recent_mean_s - m.recent_mean_s_rel).abs().max():.2e}')

# ---------------------------------------------------------------- cobertura y distribución
rows = []
for label, d in [('train R1-7', tr), ('validation R8-9', va)]:
    g = d.gap_ahead_s
    rows.append({'population': label, 'n': len(d), 'gap_missing': int(g.isna().sum()),
                 'share_gap_lt_1s': float((g < 1).mean()), 'share_gap_lt_2s': float((g < 2).mean()),
                 'share_gap_ge_cap': float((g >= GAP_CAP_S).mean())})
coverage = pd.DataFrame(rows)
by_round = (tr.assign(lt1=tr.gap_ahead_s < 1).groupby('round')
              .agg(n=('pair_id', 'size'), share_gap_lt_1s=('lt1', 'mean'),
                   median_gap_s=('gap_ahead_s', 'median')).reset_index())
dist = tr.gap_ahead_s.describe(percentiles=[.1, .25, .5, .75, .9]).rename('train_gap_ahead_s')

OUT.mkdir(exist_ok=True)
coverage.to_csv(OUT / 'feasibility_traffic_coverage.csv', index=False)
by_round.to_csv(OUT / 'feasibility_traffic_train_by_round.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'feasibility_traffic_checks.csv', index=False)
(OUT / 'feasibility_traffic_run.json').write_text(json.dumps({
    'feature': 'gap_ahead_s = end_time_s(t) - end_time_s del auto que cruzó inmediatamente antes (otro piloto, misma carrera)',
    'planned_representation': f'min(gap_ahead_s, {GAP_CAP_S})', 'validation_target_used': False,
    'models_fitted': 0, 'validation_pair_id_set_sha256': val_hash,
    'train_gap_distribution_s': dist.round(4).to_dict(), 'checks_passed': len(checks)},
    ensure_ascii=False, indent=2), encoding='utf-8')

print(coverage.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print()
print(by_round.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print(f'\n{len(checks)} checks OK · sin modelos · sin target de validación · sin test')
