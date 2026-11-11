"""Hito 1 · NextLap — reproducción de las tres referencias (mediana de train, persistencia, OLS).

Reutiliza con atribución el helper del curso `base_NextLap/nextlap_reference_v1.py`
(make_model, NUMERIC, CATEGORICAL, TARGET) y la lógica de `probar_referencia_v1.py`
del Class and Hito 1 student package v2 (IIT414W). Las verificaciones extra, las salidas
por carrera y la comparación pareada son trabajo del equipo.

Uso (desde la raíz del repo):  python hito1_reference.py
Escribe en salidas_hito1/. No lee ni existe ningún conjunto de test.
"""
from pathlib import Path
import hashlib, json, platform, sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'base_NextLap'
OUT = ROOT / 'salidas_hito1'
sys.path.insert(0, str(BASE))
from nextlap_reference_v1 import make_model, NUMERIC, CATEGORICAL, TARGET  # noqa: E402

RANDOM_SEED = 414          # OLS es determinista; se fija igual por protocolo del curso
np.random.seed(RANDOM_SEED)

# SHA-256 publicados en el reference_run.json del package v2 (versión de datos fija)
EXPECTED_SHA = {
    'train_clean_v1.csv': '43544f38cc9ef32b44886e6751e03522601265160086fe3a4956f04b87bd5c13',
    'validation_clean_v1.csv': 'b19efb6a908af12a5d2c6bdde29aa40984295e62ce0f187a201b579278db7ee8',
    'nextlap_reference_v1.py': 'f6c58925cd4a477bf3e1a5a2b3968e67df3a1cffabecee9e89701c3ac2d5f994',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = []


def check(name, ok, detail=''):
    checks.append({'check': name, 'passed': bool(ok), 'detail': str(detail)})
    if not ok:
        raise AssertionError(f'{name}: {detail}')


# ---------------------------------------------------------------- datos
hashes = {n: sha256(BASE / n) for n in EXPECTED_SHA}
check('C1 versión de datos (SHA-256)', hashes == EXPECTED_SHA, hashes)

train = pd.read_csv(BASE / 'train_clean_v1.csv')
val = pd.read_csv(BASE / 'validation_clean_v1.csv')

check('C2 conteos', (len(train), len(val)) == (5772, 1808), f'train={len(train)} val={len(val)}')
check('C3 rondas', set(train['round']) == set(range(1, 8)) and set(val['round']) == {8, 9},
      f"train={sorted(set(train['round']))} val={sorted(set(val['round']))}")
check('C4 pair_id único y disjunto',
      train.pair_id.is_unique and val.pair_id.is_unique and not set(train.pair_id) & set(val.pair_id),
      'train∩val = ∅')
check('C5 split temporal', pd.to_datetime(train.event_date).max() < pd.to_datetime(val.event_date).min(),
      f'último train {train.event_date.max()} < primer val {val.event_date.min()}')
check('C6 solo pares clean', train.clean_pair.all() and val.clean_pair.all(), 'clean_pair == True en ambos')

features = NUMERIC + CATEGORICAL
check('C7 sin columnas futuras como input',
      TARGET not in features and not any(c.startswith('next_') for c in features), features)

# ---------------------------------------------------------------- modelos
train_median = float(train[TARGET].median())
ols = make_model('OLS').fit(train[features], train[TARGET])

# El Pipeline aprende imputación/escalado/one-hot SOLO de train: el scaler debe tener la media de train.
scaler = ols.named_steps['prepare'].named_transformers_['numeric'].named_steps['scale']
train_num = train[NUMERIC].fillna(train[NUMERIC].median())
both_num = pd.concat([train, val])[NUMERIC]
check('C8 preprocesamiento ajustado solo con train',
      np.allclose(scaler.mean_, train_num.mean().to_numpy())
      and not np.allclose(scaler.mean_, both_num.mean().to_numpy()),
      dict(zip(NUMERIC, np.round(scaler.mean_, 4))))

preds = pd.DataFrame({
    'pair_id': val.pair_id, 'round': val['round'], 'event_name': val.event_name,
    'y_true_s': val[TARGET],
    'train_median': train_median,
    'persistence': val.current_lap_s.to_numpy(),
    'OLS_reference': ols.predict(val[features]),
})
models = ['train_median', 'persistence', 'OLS_reference']
check('C9 predicciones finitas, mismas filas para los 3 modelos',
      np.isfinite(preds[models].to_numpy()).all() and preds.pair_id.equals(val.pair_id),
      f'{len(preds)} filas, 1 predicción por modelo y pair_id')

# ---------------------------------------------------------------- métricas
rows = []
pops = [('all_validation', np.ones(len(val), bool))] + [
    (f"round_{r} ({val.loc[val['round'] == r, 'event_name'].iat[0]})", (val['round'] == r).to_numpy())
    for r in sorted(val['round'].unique())]
for m in models:
    for label, mask in pops:
        rows.append({'model': m, 'population': label, 'n': int(mask.sum()),
                     'MAE_s': mean_absolute_error(preds.y_true_s[mask], preds[m][mask])})
metrics = pd.DataFrame(rows)

# Comparación pareada OLS vs persistencia (mismas filas)
err_p = (preds.y_true_s - preds.persistence).abs()
err_o = (preds.y_true_s - preds.OLS_reference).abs()
paired = []
for label, mask in pops:
    d = (err_o - err_p)[mask]
    paired.append({'population': label, 'n': int(mask.sum()),
                   'MAE_OLS_minus_persistence_s': d.mean(),
                   'share_rows_OLS_better': float((d < 0).mean())})
paired = pd.DataFrame(paired)

coef = pd.DataFrame({'encoded_input': ols.named_steps['prepare'].get_feature_names_out(),
                     'coefficient': ols.named_steps['regression'].coef_})

# ---------------------------------------------------------------- salidas
OUT.mkdir(exist_ok=True)
metrics.to_csv(OUT / 'reference_metrics.csv', index=False)
paired.to_csv(OUT / 'paired_ols_vs_persistence.csv', index=False)
preds.to_csv(OUT / 'validation_reference_predictions.csv', index=False)
coef.to_csv(OUT / 'ols_coefficients.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'checks.csv', index=False)
val_ids_hash = hashlib.sha256('\n'.join(sorted(val.pair_id)).encode()).hexdigest()
run = {
    'train_rows': len(train), 'validation_rows': len(val), 'test_used': False, 'seed': RANDOM_SEED,
    'features': features, 'target': TARGET, 'train_median_s': train_median,
    'validation_pair_id_set_sha256': val_ids_hash,
    'data_sha256': hashes,
    'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                    'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                    'platform': platform.platform()},
    'checks_passed': len(checks),
}
(OUT / 'reference_run.json').write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding='utf-8')

pd.set_option('display.width', 120)
print(metrics.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print()
print(paired.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print(f'\n{len(checks)} checks OK · mediana train = {train_median:.3f} s · '
      f'hash pair_id validación = {val_ids_hash[:16]}… · sin test')
