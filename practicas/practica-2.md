# Práctica 2 — Artefactos, experimentos y MLflow

**Objetivo:** guardar y recargar el pipeline con garantías, registrar y comparar
ejecuciones en MLflow y hacer una búsqueda de hiperparámetros **acotada** sin usar
el conjunto de test para elegir.

**Duración:** 150 min · **Requisito:** práctica 1 terminada (o checkpoint de la
sesión 1).

## Qué debe quedar hecho

- [ ] Run *baseline* en MLflow con parámetros, métricas y modelo.
- [ ] Run de búsqueda (`--busqueda`) con el mejor resultado de validación cruzada.
- [ ] Modelo registrado (`churn-abandono`) con al menos dos versiones.
- [ ] `artifacts/metadata.json` con `model_version` igual a la versión registrada.
- [ ] Commit al terminar.

## Paso 1 — Parámetros aprendidos vs hiperparámetros (0–15 min)

Abre `artifacts/model.joblib` en un notebook o en `uv run python`:

```python
import joblib

m = joblib.load("artifacts/model.joblib")
m.named_steps["clasificador"].coef_  # aprendido en fit
m.named_steps["clasificador"].get_params()  # hiperparámetros (configuración)
```

## Paso 2 — Guardar, recargar y verificar (15–35 min)

Completa el `TODO S2` del final de `main()` en `train.py`: recargar el artefacto y
comprobar con `np.allclose` que predice **igual** que el modelo en memoria.

Recuerda: `joblib` solo para artefactos **propios** y en el **mismo entorno**
(mismas versiones de librerías, por eso el `uv.lock`).

## Paso 3 — MLflow local (35–55 min)

1. Completa `train.py` → `log_to_mlflow` (`TODO S2`) y sustituye en `main()` la línea
   de la versión local por la llamada a `log_to_mlflow(...)`.
2. Entrena: `uv run python -m mlops_course.train`

   ```text
   Successfully registered model 'churn-abandono'.
   Created version '1' of model 'churn-abandono'.
   …
   Run <run_id> registrado como versión 1
   ```

3. En **otra terminal**: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`
   y abre <http://127.0.0.1:5001>. Localiza el experimento `churn-abandono`, el run
   `baseline`, sus parámetros, métricas y el modelo registrado.

## Paso 4 — Búsqueda de hiperparámetros (65–95 min)

Completa `train.py` → `search` (`TODO S2`) con `GridSearchCV`:

- Espacio pequeño definido en `configs/dev.yaml` (`busqueda.param_grid`: 8
  combinaciones × 5 folds).
- Se ajusta **solo con train**. Validación y test se evalúan después con el
  modelo ya elegido.

```bash
uv run python -m mlops_course.train --busqueda
```

Salida orientativa:

```text
Mejor roc_auc en CV: 0.83…
Mejores parámetros: {'clasificador__C': …, 'clasificador__class_weight': …}
…
Run … registrado como versión 2
```

**Pregunta clave:** ¿por qué no elegimos los hiperparámetros mirando
`test_roc_auc`?

## Paso 5 — Comparar y recuperar (95–120 min)

- En MLflow, selecciona los dos runs → *Compare*. ¿Qué cambió? ¿Mejoró `val_roc_auc`?
  ¿Y `val_recall` con `class_weight=balanced`?
- Completa `predict.py` → `load_registered_model` (`TODO S2`) y comprueba:

  ```python
  from mlops_course.config import load_config
  from mlops_course.predict import load_registered_model
  import pandas as pd

  m = load_registered_model(load_config(), "1")
  m.predict_proba(pd.read_csv("data/churn_nuevos.csv").head(3))
  ```

## Paso 6 — Demo opcional de Optuna (120–140 min)

La hace el docente. Si quieres reproducirla: `uv sync --group optuna`. No es
requisito ni puntúa en la rúbrica.

## Cierre (140–150 min)

Responde en tu README (sección "Reproducibilidad"): ¿qué necesita otra persona
para obtener **el mismo** modelo? (código + commit, `uv.lock`, configuración,
datos, semilla…). `git commit -m "S2: tracking y registro en MLflow"`.

> Si usas `random_forest`, adapta `busqueda.param_grid` a sus hiperparámetros
> (p. ej. `clasificador__max_depth: [5, 10, null]`).
