# Práctica 3 — Inferencia batch, FastAPI e interfaz Streamlit

**Objetivo:** reutilizar el **mismo** modelo en batch y online (API), y añadir una
interfaz que **consuma la API por HTTP** sin duplicar la inferencia.

```text
Streamlit (ui.py) ──HTTP POST /predict──▶ FastAPI (api.py) ──▶ modelo (artifacts/)
      formulario y visualización              contrato + inferencia
```

**Duración:** 150 min · **Requisito:** práctica 2 terminada (modelo en
`artifacts/` con versión de MLflow).

## Qué debe quedar hecho

- [ ] `reports/predicciones.csv` con `probabilidad`, `abandona`, `umbral` y
      `model_version`.
- [ ] API con `GET /health` y `POST /predict` que valida la entrada (422 si no
      cumple el contrato).
- [ ] UI que envía el formulario a la API y muestra probabilidad, clase, umbral y
      versión, con mensajes de error claros.
- [ ] Captura de la UI y comandos de arranque en tu README. Commit.

## Contrato

**Entrada** (`schemas.Cliente`): `antiguedad_meses` (0–120), `cuota_mensual`
(> 0 y ≤ 500), `tipo_contrato`, `tipo_internet`, `metodo_pago` (valores en
`data/README.md`). Campos extra → 422.

**Salida** (`schemas.Prediccion`):

```json
{"abandona": 1, "probabilidad": 0.73, "umbral": 0.5, "model_version": "2"}
```

## Paso 1 — Batch (10–30 min)

```bash
uv run python -m mlops_course.predict
```

Abre `reports/predicciones.csv`: ¿qué versión de modelo aparece? ¿Coincide con
`artifacts/metadata.json` y con MLflow?

## Paso 2 — API (30–60 min)

Completa los `TODO S3` de:

1. `schemas.py` → los cuatro campos restantes de `Cliente`.
2. `api.py` → `lifespan`: cargar el modelo **una vez** al arrancar.
3. `api.py` → `predict`: 503 si no hay modelo; predecir con `predict_df` y
   devolver `Prediccion`.

```bash
uv run uvicorn mlops_course.api:app --host 127.0.0.1 --port 8000
```

## Paso 3 — Probar el contrato (70–90 min)

Abre <http://127.0.0.1:8000/docs> → `POST /predict` → *Try it out* (el ejemplo ya
viene relleno). Después, desde otra terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"antiguedad_meses": 2, "cuota_mensual": 95.5, "tipo_contrato": "mensual", "tipo_internet": "fibra", "metodo_pago": "cheque_electronico"}'
# Entrada inválida -> HTTP 422
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"antiguedad_meses": 2, "cuota_mensual": -1, "tipo_contrato": "mensual", "tipo_internet": "fibra", "metodo_pago": "cheque_electronico"}'
```

> En Windows PowerShell usa `curl.exe` o `Invoke-RestMethod`, o simplemente `/docs`.

## Paso 4 — Interfaz Streamlit (90–125 min)

El formulario ya está hecho en `ui.py`. Completa el `TODO S3`: llamada HTTP con
`timeout`, manejo de errores **sin inventar predicciones** y visualización de la
respuesta validada con `Prediccion`.

Con la API arrancada, en otra terminal:

```bash
uv run streamlit run src/mlops_course/ui.py --server.address 127.0.0.1 --server.port 8501
```

Abre <http://127.0.0.1:8501>.

## Paso 5 — Comprobaciones (125–140 min)

1. **Coherencia:** introduce en la UI la primera fila de `data/churn_nuevos.csv`.
   La probabilidad debe coincidir con `/docs` y con `reports/predicciones.csv`
   (tolerancia 1e-6).
2. Cambiar campos **no** llama a la API hasta pulsar «Predecir» (`st.form`).
3. **Servicio caído:** para la API (`Ctrl+C`) y vuelve a predecir → mensaje de
   servicio no disponible, sin mostrar la predicción anterior.
4. **422 desde la UI:** los widgets impiden valores inválidos; el docente mostrará
   un servicio simulado que devuelve 422, 500, timeout y JSON incompatible.
   No quites la validación del servicio.

## Cierre

Añade a tu README los comandos de arranque y una captura de la UI.
`git commit -m "S3: batch, API y UI"`. Ejecuta antes `uv run ruff check .`.

**No se pide:** desplegar en internet, autenticación, Docker ni ONNX (demos
opcionales del docente).
