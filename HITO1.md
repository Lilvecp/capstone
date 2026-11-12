# Hito 1 · NextLap · Framing and experiment plan

Equipo: Castro–Bufadel (sin nombre formal) · Integrantes: Valentín Castro, Camilo Bufadel
Repo: https://github.com/Lilvecp/capstone · Commit: ver tag `Hito1` y el texto de entrega · Fecha: 12-11-2026

## 1. Problem and decision value  — [A · Valentín]

| Elemento | Definición |
|---|---|
| Target y unidad | `next_lap_s`: duración de la vuelta t+1 del **mismo piloto**, en **segundos**. |
| Momento de predicción | Al terminar la vuelta t (`end_time_s` de t), usando solo información observada hasta ese instante y dentro del mismo piloto/carrera/stint. |
| Lector previsto | Ingeniero de carrera / estratega en el muro de boxes que sigue el ritmo de su piloto durante un stint. |
| Decisión que informa | Fijar un **ritmo esperado para la próxima vuelta**. Si la vuelta real se desvía bastante más que el error típico (~0,4 s), es una señal para revisar la causa (degradación acelerada, tráfico, problema del auto) y decidir si hablar con el piloto o reevaluar la ventana de parada. |
| Qué NO establece | No dice **cuándo** conviene entrar a boxes, cuántas posiciones ganaría una estrategia ni el efecto causal del neumático: solo estima un tiempo de vuelta condicionado a que t+1 sea una vuelta limpia. |

**Límite de la población `clean_pair`.** `clean_pair` se define en retrospectiva con condiciones de t+1: vuelta precisa, sin entrada ni salida de boxes, no borrada, bandera verde (`track_status = 1`) y mismo stint/compuesto. En vivo no se sabe de antemano si t+1 será limpia. En las carreras de validación, solo el 82,4 % (R8) y el 71,8 % (R9) de los pares consecutivos observados son `clean_pair` (77 % en el total de R1–9: 7.580 de 9.849). Los MAE de este informe valen **solo para esa población condicional** y no son evidencia de desempeño en carrera completa ni en tiempo real. Las vueltas con safety car, pit o tráfico fuerte quedan fuera y ahí se espera un error mayor.

## 2. Reproduced reference  — [A · Valentín]

| Elemento | Valor |
|---|---|
| Datos | `IIT414W_Capstone_NextLap_StudentData_v1` (idéntico a `base_NextLap/` del package v2). SHA-256 verificados en el check C1: train `43544f38…`, validación `b19efb6a…`, helper `f6c58925…` |
| Split | Train: rondas 1–7 (Australia → Barcelona), **n = 5.772**. Validación: rondas 8–9 (Austria, Gran Bretaña), **n = 1.808**. Ningún `pair_id` compartido; el último evento de train (14-06) es anterior al primero de validación (28-06). Seed 414. Sin test. |
| Features | `current_lap_s`, `recent_mean_s`, `current_lap_number`, `current_tyre_life` (numéricas); `current_compound`, `current_team` (categóricas). Ninguna `next_*` (check C7). |
| Fit / transform | `make_model('OLS')` es un `Pipeline`: imputación por mediana + `StandardScaler` (numéricas) y moda + one-hot (categóricas), luego `LinearRegression`. `.fit` se llama **solo con train**; validación solo pasa por `.predict`. El check C8 confirma que las medias del scaler son las de train y no las de train+validación. La mediana de referencia (84,762 s) se calcula solo con train. |
| Comando | `python hito1_reference.py` → `salidas_hito1/` (métricas, predicciones por `pair_id`, checks, entorno). Reutiliza con atribución el helper del curso. |

**Resultados en validación, mismas 1.808 filas para los tres modelos (MAE en segundos):**

| Modelo | Toda la validación (n = 1.808) | R8 Austria (n = 1.066) | R9 Gran Bretaña (n = 742) |
|---|---|---|---|
| Mediana de train (84,762 s constante) | 11,376 | 11,716 | 10,888 |
| Persistencia (`next = current_lap_s`) | 0,409 | 0,401 | 0,422 |
| OLS referencia (6 inputs) | **0,387** | **0,374** | **0,406** |

Fuente: `salidas_hito1/reference_metrics.csv`. Hash del conjunto de `pair_id` de validación: `f453a98dc906904c…` (en `reference_run.json`), que sirve para comprobar en el Hito 2 que se comparan las mismas filas.

**Interpretación (OLS vs persistencia).** OLS baja el MAE en **0,023 s (−5,5 %)** frente a repetir la vuelta actual, y la mejora aparece en las dos carreras (−0,027 s en R8 y −0,016 s en R9). La ganancia es chica y poco uniforme: OLS tiene menor error absoluto en solo el **51,6 %** de las filas (49,6 % en R9), así que su ventaja viene de corregir menos casos con error grande y no de ser mejor vuelta a vuelta (`paired_ols_vs_persistence.csv`). Los coeficientes estandarizados más grandes son `recent_mean_s` (4,29) y `current_lap_s` (3,49), es decir, OLS es básicamente una persistencia suavizada. La mediana de train falla por ~11 s porque el nivel de tiempo de vuelta depende del circuito (73–109 s en train). Con dos carreras de validación, la diferencia es evidencia de selección, no una conclusión general.

*Nota de ejecución:* ver el entorno exacto (Python/numpy/pandas/sklearn) en `salidas_hito1/reference_run.json`. Los MAE coinciden con las salidas del instructor hasta ~1e-15 s.

## 3. Primary hypothesis  — [B · Camilo]

Dirección elegida: **Tráfico**, medido con los propios datos de tiempos publicados (sin fuente externa).

| Elemento | Definición |
|---|---|
| Feature propuesta | `gap_ahead_s` = `end_time_s` de la vuelta t del piloto − `end_time_s` del auto (otro piloto, misma carrera) que cruzó la línea **inmediatamente antes**. Representación pre-declarada: `gap_ahead_cap_s = min(gap_ahead_s, 5,0)`. Se agrega como **una** numérica más a los 6 inputs de la referencia. |
| Mecanismo esperado | Un auto a < ~1–2 s adelante al final de t probablemente sigue cerca en t+1: aire sucio (más lento, sobre todo en curvas) o rebufo/DRS (más rápido en rectas). `current_lap_s` y `recent_mean_s` ya reflejan el tráfico de t, pero no dicen si el piloto **sigue** detrás de alguien; la feature agrega esa información. Más allá de ~5 s el efecto debería ser nulo, por eso el tope. No fijamos el signo de antemano: lo que se evalúa es si mejora el MAE frente a la misma referencia sin ella. |
| Fuente | `base_NextLap/data/race_2026_r0X_laps_v2.csv` (release v1, SHA-256 verificado contra `source_manifest_v2.json` por `load_snapshots`). Están las 21–22 vueltas por carrera de **todos** los pilotos, con un mismo reloj de sesión `end_time_s`. |
| Disponibilidad en t | Solo usa `end_time_s` de otros autos con `end_time_s ≤ end_time_s(t)`: ya cruzaron la línea cuando termina t, así que es observable en ese instante. No usa vueltas futuras, gaps posteriores, posición final ni ningún `next_*`. Se calcula sobre la **línea de tiempo original completa** (incluye vueltas de pit, SC y no precisas de los demás, porque el tráfico físico existe igual) **antes** del filtro `clean_pair`, y después se une por (`round`, `driver_number`, `current_lap_number`). |
| Verificación ya hecha | `python hito1_feasibility_traffic.py` → 7 checks OK (`salidas_hito1/feasibility_traffic_*`): el timestamp usado nunca es posterior al fin de t (F2), nunca es el mismo piloto (F3), el set de `pair_id` clean reconstruido es **idéntico** al del release en train (5.772) y validación (1.808) (F4–F6, hash `f453a98d…`) y `recent_mean_s` se reconstruye igual (F7). Cobertura: **0 faltantes** en train y validación. En train, el 22,9 % de los pares tiene un auto a < 1 s y el 27,9 % está a ≥ 5 s (en validación, 18,1 % y 24,9 %). El script no ajusta modelos ni lee el target de validación. |
| Datos faltantes / historial corto | La feature no depende de historial del stint, así que no hay regla de ventana corta. Solo puede faltar si nadie cruzó antes en la sesión (no ocurre con vuelta ≥ 2 en R1–9). Regla fija: si falta, se usa el tope (5 s = "aire libre") y se cuenta cuántas filas la usaron. Esto debe dar 0 filas en validación. |
| Riesgos conocidos | (i) El auto "anterior" puede ser uno con una vuelta de diferencia o uno que está entrando a boxes: igual es tráfico en pista, salvo en el caso de pit. Lo cubrimos con la sensibilidad S1 (§4). (ii) Que haya un auto cerca al final de t no garantiza que siga cerca en t+1. (iii) La población `clean_pair` (verde, sin pit) ya excluye parte del tráfico más extremo. |
| Plan B | No se necesita fuente externa. Si la feature no aporta, el resultado negativo se reporta igual. Como representación alternativa ya declarada queda el indicador `gap_ahead_s < 1,0` (zona DRS), que también se calcula con los datos publicados. |

## 4. Experiment plan  — [B · Camilo]

**Fijo en todas las corridas:** target `next_lap_s` (s); split R1–7 / R8–9; las **mismas 1.808 filas** de validación (hash del set de `pair_id` = `f453a98d…`, que se verifica antes de calcular el MAE); el mismo modelo `make_model('OLS', numeric=NUMERIC+[feature])` del helper; preprocesamiento (imputación por mediana + escalado + one-hot) ajustado **solo con train** dentro del `Pipeline`; métrica MAE en segundos; seed 414.

| Exp. | Inputs | Modelo | Filas | Qué aísla |
|---|---|---|---|---|
| E0 Referencia | 6 inputs de referencia | OLS | train 5.772 / val 1.808 | Línea base (MAE 0,387 s, ya reproducida) |
| E1 Extensión | 6 + `gap_ahead_cap_s` | OLS (mismo) | mismas | E1 − E0 = aporte de la información de tráfico |
| E2 Placebo | 6 + `gap_ahead_cap_s` **permutado al azar dentro de cada carrera** (seed 414) | OLS (mismo) | mismas | Que la mejora no venga solo de tener una columna más con esa distribución. Si E2 ≈ E1, la mejora no es por el tráfico |
| E1b Alternativa pre-declarada | 6 + indicador `gap < 1 s` | OLS (mismo) | mismas | Representación alternativa (solo si se elige en la selección) |
| S1 Sensibilidad | igual a E1, pero sin contar como "auto adelante" a los que están en vuelta de pit in/out | OLS (mismo) | mismas | Si la señal depende de autos que están en boxes |

La ablación principal es **E1 vs E0** (la misma extensión sin la información añadida = la referencia), con el **placebo E2** como control de que no es solo "una columna más". No se cambia de algoritmo: cambiar el modelo no prueba si la información sirve.

**Selección vs evaluación final.**
1. La elección entre E1 y E1b (como máximo **2 variantes**, el tope de 5 s fijado antes) se hace con *leave-one-race-out* **dentro de train** (7 folds, R1–7). Se elige la de menor MAE medio. Si la diferencia es < 0,005 s, se queda E1 por ser más simple.
2. **Validación R8–9 se usa una sola vez** para la variante elegida, su placebo y S1. No se vuelve a ajustar nada después de mirarla.
3. El **test reservado** no está en el paquete y no se usa para elegir features, parámetros ni modelos. Ningún integrante tuvo acceso previo a él.
4. Todas las corridas, incluidas las que no mejoran, quedan en `DECISION_LOG.md` y en un `trials_log.csv` (variante, folds, MAE, fecha y quién).

**Qué apoyaría / contradiría la hipótesis (fijado antes de implementar):**
- *Apoya:* MAE(E1) < MAE(E0) en validación **y en las dos carreras por separado (R8 y R9)**, E2 sin mejora (|E2 − E0| < la mitad de la ganancia de E1), y una mejora concentrada en filas con `gap < 2 s` (si aparece donde el gap es ≥ 5 s, sería sospechosa). Se reporta la diferencia pareada E1 − E0 por fila con un IC bootstrap 95 % por bloques de piloto-carrera.
- *Contradice:* E1 ≥ E0, una mejora solo en una carrera, o E2 ≈ E1. Cualquiera de estos se reporta como resultado negativo sin cambiar el plan.
- *No establece:* aunque mejore, no prueba un efecto causal del aire sucio ni sirve para decidir adelantamientos o paradas. Vale solo para la población `clean_pair` de R8–9.

**Checks escritos antes de implementar (Hito 2):**

| Check | Evidencia | Falla si |
|---|---|---|
| K1 Temporal | `assert` de que el `end_time_s` del auto usado es ≤ `end_time_s(t)` en cada fila (como F2) | algún timestamp es posterior al fin de t |
| K2 Mismas filas | hash del set de `pair_id` de validación de E0/E1/E2 = `f453a98d…` y mismo orden | cambia el hash o el n ≠ 1.808 |
| K3 Sin fuga | ninguna columna `next_*` ni `next_lap_s` entre los inputs (`make_model` lo bloquea) y la feature se calcula sin `clean_pair` | aparece una columna futura o la feature cambia al filtrar |
| K4 Fit solo en train | la media del scaler para `gap_ahead_cap_s` = media de train (como C8) | coincide con la media de train+val |
| K5 Faltantes | se cuentan las filas imputadas con el tope | > 0 en validación sin explicación |
| K6 Placebo | la permutación se hace dentro de la carrera, con seed 414 y la misma distribución marginal | cambia la distribución o se usa otra seed |

**Robustez (plan):** MAE por carrera de validación con conteos (R8 n = 1.066, R9 n = 742) para E0/E1/E2. MAE por tramo de gap (< 1 s, 1–2 s, 2–5 s, ≥ 5 s) con n de cada tramo. Análisis en la población amplia (`observed_pairs_train_validation_v1.csv`), reportado **aparte** de la comparación común. Errores por contexto y SHAP en el Hito 2.

## 5. Traceability and next steps  — [ambos]

| Integrante | Decisiones/partes propias | Puede explicar |
|---|---|---|
| Valentín Castro | §1 encuadre y límite `clean_pair`, §2 reproducción con 9 checks, `hito1_reference.py`, README y empaquetado | Por qué el scaler se ajusta solo con train (C8), por qué OLS gana solo en el 51,6 % de las filas, el porcentaje de `clean_pair` por carrera |
| Camilo Bufadel | §3 hipótesis de tráfico, §4 plan de experimento, `hito1_feasibility_traffic.py`, segunda reproducción de la referencia en otra máquina | Por qué `gap_ahead_s` está disponible en t (F2), por qué se calcula antes del filtro `clean_pair`, para qué sirve el placebo, por qué la selección se hace con LORO en train y no con validación |

- **Registro de IA:** ver `DECISION_LOG.md` (quién, para qué, qué se adoptó y cómo se verificó).
- **Revisión tras feedback:** ver `DECISION_LOG.md` → "Revisión tras feedback".
- **Antes del Hito 2 (19-11):** Camilo implementa E1/E2/S1 y los checks K1–K6 y hace la selección LORO en train. Valentín agrega MAE por carrera y por tramo de gap, IC bootstrap y SHAP. Ambos registran todos los intentos en `trials_log.csv`.
- **Bloqueo conocido:** ninguno técnico. Pendiente confirmar con Ángel (12–13/11) si el auto que va en pit lane cruzando la línea debe contar como tráfico (es la sensibilidad S1).
