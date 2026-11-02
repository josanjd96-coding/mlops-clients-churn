# Proyecto de prácticas MLOps — abandono de clientes (churn)

Entorno de trabajo para las **cuatro prácticas** del módulo de MLOps. A lo largo de
las sesiones convertirás un notebook en un proyecto reproducible con experimentos
trazables, inferencia batch, una API, una interfaz y un análisis de drift.

```text
notebook ──▶ src/ (S1) ──▶ artefacto + MLflow (S2) ──▶ batch · API · UI (S3) ──▶ drift (S4)
```

> **Antes de la primera sesión:** sigue [`SETUP.md`](SETUP.md) y ejecuta el
> *smoke test*. Si algo falla, avisa por el canal de dudas **antes** de clase.

## Proyecto final común: Heart Disease

Las prácticas guiadas usan Churn. La entrega final es **por parejas** y utiliza
Heart Disease para todo el grupo: [enunciado y rúbrica](proyecto-final.md).
El CSV está incluido en [`data/alt/heart_disease.csv`](data/alt/heart_disease.csv),
con [diccionario, licencia y ejemplo de carga](data/alt/README.md).
No hace falta descargarlo. El notebook `alt_heart_disease.ipynb` es un punto de
partida; adaptar configuración, validación y contrato de API es parte del trabajo.

## Estructura

```text
.
├── README.md               # este fichero: comandos y mapa del proyecto
├── SETUP.md                # instalación paso a paso y checklist de preflight
├── pyproject.toml          # dependencias con versiones fijadas
├── uv.lock                 # lock probado: NO lo modifiques
├── Makefile                # atajos opcionales (equivalencias abajo)
├── configs/                # configuración no sensible por entorno
│   ├── dev.yaml            #   la que se usa por defecto
│   ├── preprod.yaml        #   ejemplo de otro entorno
│   └── example.yaml        #   plantilla documentada
├── data/                   # datos locales (ver data/README.md)
├── notebooks/
│   ├── baseline.ipynb      # punto de partida de la práctica 1
│   └── alt_heart_disease.ipynb
├── practicas/              # enunciados: practica-1.md … practica-4.md
├── scripts/smoke_test.py   # comprueba el entorno
├── src/mlops_course/       # el código del proyecto (con TODO a completar)
│   ├── config.py           #   carga configs/<entorno>.yaml
│   ├── data.py             #   S1 carga, validación y split
│   ├── models.py           #   S1 pipeline (preprocesado + modelo) y factory
│   ├── train.py            #   S1 entrenamiento · S2 búsqueda y MLflow
│   ├── predict.py          #   S1 predicción y batch · S2 carga desde MLflow
│   ├── schemas.py          #   S3 contrato de entrada/salida (Pydantic)
│   ├── api.py              #   S3 API FastAPI
│   ├── ui.py               #   S3 interfaz Streamlit (cliente HTTP de la API)
│   └── monitoring.py       #   S4 drift con Evidently y comparación de modelos
├── tests/test_ejemplo.py   # bonus: tests de ejemplo
├── artifacts/              # (se genera) modelo y metadatos — ignorado por Git
└── reports/                # (se genera) predicciones e informes — ignorado por Git
```

Los fragmentos que debes completar están marcados con `TODO S<n>` (n = número de
práctica). Mientras no los completes, ese código lanza `NotImplementedError`.

## Prácticas

| Sesión | Enunciado | Resultado |
|---|---|---|
| 1 | [Del notebook al proyecto](practicas/practica-1.md) | `train` y `predict` desde terminal, lint limpio, primer commit |
| 2 | [Artefactos, experimentos y MLflow](practicas/practica-2.md) | Modelo cargable, runs comparables y modelo registrado |
| 3 | [Batch, FastAPI y Streamlit](practicas/practica-3.md) | Predicciones batch, API con contrato y UI que la consume |
| 4 | [Drift y cierre](practicas/practica-4.md) | Informe de drift, decisión razonada y entrega final |

## Comandos

Ejecútalos **siempre desde la raíz del proyecto**. `make` es opcional: la columna
de la derecha es el comando real y funciona en macOS, Linux y Windows.

| Atajo | Comando equivalente | Para qué |
|---|---|---|
| `make install` | `uv sync` | Crear `.venv` e instalar dependencias del lock |
| `make smoke-test` | `uv run python scripts/smoke_test.py` | Comprobar el entorno |
| `make lint` | `uv run ruff check .` y `uv run ruff format --check .` | Revisar estilo |
| `make format` | `uv run ruff format .` y `uv run ruff check --fix .` | Corregir estilo |
| `make train` | `uv run python -m mlops_course.train` | Entrenar el baseline |
| `make search` | `uv run python -m mlops_course.train --busqueda` | Búsqueda de hiperparámetros (S2) |
| `make mlflow` | `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001` | Abrir MLflow en <http://127.0.0.1:5001> |
| `make batch` | `uv run python -m mlops_course.predict` | Predicciones de `data/churn_nuevos.csv` |
| `make api` | `uv run uvicorn mlops_course.api:app --host 127.0.0.1 --port 8000` | API en <http://127.0.0.1:8000/docs> |
| `make ui` | `uv run streamlit run src/mlops_course/ui.py --server.address 127.0.0.1 --server.port 8501` | UI en <http://127.0.0.1:8501> |
| `make monitor` | `uv run python -m mlops_course.monitoring simular --escenario covariables`, luego `... informe` y `... comparar` | Drift (S4) |
| `make test` | `uv run pytest` | Tests de ejemplo (bonus) |
| `make clean` | borrar `artifacts/ reports/ mlruns/ mlartifacts/ mlflow.db` | Empezar de cero |

Notebooks: `uv run jupyter lab` (o abre la carpeta en VS Code y elige el intérprete
`.venv`).

### Entornos y variables

| Variable | Por defecto | Uso |
|---|---|---|
| `APP_ENV` | `dev` | Elige `configs/<APP_ENV>.yaml` (también `--entorno preprod`) |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Sobrescribe el tracking de MLflow |
| `API_URL` | `http://127.0.0.1:8000` | Dirección de la API que usa la UI |

Ejemplo: `APP_ENV=preprod uv run python -m mlops_course.train` (en PowerShell:
`$env:APP_ENV="preprod"; uv run python -m mlops_course.train`).

## Reglas del proyecto

- **No subas** a Git: `.venv/`, `artifacts/`, `reports/`, `mlflow.db`, `mlruns/`,
  secretos ni datos personales (ya están en `.gitignore`).
- **No modifiques** `uv.lock` ni instales paquetes con `pip` durante las sesiones.
- Los artefactos `joblib` solo se cargan si los has generado tú: deserializar un
  fichero desconocido puede ejecutar código.
- Haz un **commit al final de cada bloque** de la práctica (checkpoint).

## Si te quedas atrás

Pide al docente el *checkpoint* de la sesión anterior: es este mismo proyecto con
los TODO de esa sesión resueltos. Copia tu carpeta `notebooks/` si la has
modificado y continúa desde el checkpoint.
