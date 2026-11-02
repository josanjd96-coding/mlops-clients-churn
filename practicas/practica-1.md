# Práctica 1 — Del notebook al proyecto

**Objetivo:** que el modelo del notebook se pueda entrenar y usar **desde la
terminal**, con código organizado, configuración separada y estilo revisado.

**Duración:** 150 min · **Requisito:** smoke test en verde (ver `SETUP.md`).

## Qué debe quedar hecho

- [ ] `uv run python -m mlops_course.train` entrena, evalúa y guarda
      `artifacts/model.joblib` y `artifacts/metadata.json`.
- [ ] `uv run python -m mlops_course.predict` genera `reports/predicciones.csv`.
- [ ] Los parámetros (rutas, semilla, umbral, hiperparámetros) viven en
      `configs/dev.yaml`, no en el código.
- [ ] `uv run ruff check .` y `uv run ruff format --check .` sin avisos.
- [ ] Al menos un commit con un mensaje que explique el cambio.

## Paso 1 — Entender el notebook (15–35 min)

Abre `notebooks/baseline.ipynb` y ejecútalo con *Restart Kernel and Run All*.
Anota en una tabla:

| Pieza | ¿Dónde está en el notebook? | ¿Dónde irá en el proyecto? |
|---|---|---|
| Entradas y target | | `data.py` (`FEATURES`, `TARGET`) |
| Lectura y validación de datos | | `data.py` → `load_data` |
| Split train/validación/test | | `data.py` → `split_data` |
| Preprocesamiento + modelo | | `models.py` → `build_pipeline` |
| Evaluación | | `train.py` → `evaluate` |
| Guardado del modelo | (no existe) | `train.py` → `save_artifacts` |
| Predicción | | `predict.py` → `predict_df`, `batch_predict` |
| Valores fijos (semilla, rutas…) | | `configs/dev.yaml` |

## Paso 2 — Completar los TODO de la sesión 1 (35–60 min y 70–90 min)

Busca `TODO S1` en `src/mlops_course/` (en VS Code: `Ctrl/Cmd+Shift+F`). En orden:

1. **`data.py` → `load_data`**: leer el CSV y lanzar `ValueError` si falta alguna
   columna de `FEATURES + [TARGET]`.
2. **`data.py` → `split_data`**: dos `train_test_split` estratificados con la
   semilla de la configuración (primero test, luego validación).
3. **`models.py` → `build_pipeline`**: el `ColumnTransformer` + clasificador del
   notebook, usando la *factory* `build_estimator`.
4. **`train.py` → `evaluate`**: `roc_auc`, `f1`, `precision`, `recall`, `accuracy`
   aplicando el umbral a `predict_proba`.
5. **`train.py` → `save_artifacts`**: `joblib.dump` del pipeline y `json.dump`
   de los metadatos.
6. **`predict.py` → `load_model`, `predict_df`, `batch_predict`**.

Ejecuta tras cada bloque: `uv run python -m mlops_course.train`. El error te dirá
cuál es el siguiente `TODO` pendiente.

**Salida esperada** (valores orientativos, pueden variar ligeramente):

```text
Filas -> train: 4507, validación: 1127, test: 1409
  val_roc_auc: 0.84…
  …
  test_roc_auc: 0.83…
Modelo guardado en …/artifacts/model.joblib
```

Y con `uv run python -m mlops_course.predict`:

```text
50 predicciones guardadas en …/reports/predicciones.csv
Versión del modelo: local-AAAAMMDD-HHMMSS
```

> En esta sesión la versión es una marca de tiempo local. En la práctica 2 pasará
> a ser la versión registrada en MLflow.

## Paso 3 — Configuración y entornos (70–90 min)

- Revisa `configs/dev.yaml` y `config.py`: ¿qué pasa si ejecutas
  `uv run python -m mlops_course.train --entorno preprod`? ¿Dónde se guarda el modelo?
- Cambia `umbral` a `0.3` en `dev.yaml`, reentrena y compara `recall`/`precision`.
  Vuelve a dejarlo en `0.5`.
- ¿Dónde pondrías una contraseña de base de datos? (Respuesta: en una variable de
  entorno, nunca en el YAML ni en Git.)

## Paso 4 — Estilo con Ruff (90–110 min)

```bash
uv run ruff check .          # problemas (imports sin usar, nombres no definidos…)
uv run ruff format --check . # ¿está formateado?
uv run ruff format .         # formatear
uv run ruff check --fix .    # corregir lo automático
```

Ruff no sustituye a los tests: un código bien formateado puede seguir siendo
incorrecto.

## Paso 5 — Git y reproducibilidad (110–130 min)

```bash
git status            # ¿aparecen artifacts/ o .venv/? (no deberían: .gitignore)
git add src configs
git commit -m "S1: pipeline, entrenamiento y predicción desde src/"
```

Prueba de reproducibilidad: `make clean` (o borra `artifacts/` y `reports/`) y
vuelve a ejecutar `train` y `predict` desde cero. Las métricas deben coincidir.

## Bonus (130–145 min)

- `uv run pytest`: los tests de `tests/test_ejemplo.py` que dependen de la sesión 1
  deberían pasar ahora.
- Cambia `modelo.tipo` a `random_forest` (con `parametros: {n_estimators: 200,
  random_state: 42}`) y compara. Mira cómo la *factory* evita tocar el código.

## Antes de la práctica 2

- Commit hecho y `train` funcionando desde un proceso limpio.
- Si no has terminado, pide el checkpoint de la sesión 1.
