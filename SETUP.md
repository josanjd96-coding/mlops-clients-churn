# SETUP — preparar el entorno antes de la práctica 1

Objetivo: llegar a la primera sesión con el entorno **funcionando y comprobado**.
Tiempo estimado: 20–30 minutos (la mayor parte es descarga).

Ruta oficial del curso: **Python 3.12 + [uv](https://docs.astral.sh/uv/)** con el
lock incluido (`uv.lock`). No uses conda, `pip install` suelto ni otras versiones de
Python: así todos tenemos exactamente las mismas librerías.

## 1. Instalar las herramientas

### Git

- **macOS:** `xcode-select --install` (o `brew install git`).
- **Windows:** <https://git-scm.com/download/win> (deja las opciones por defecto).
- **Linux:** `sudo apt install git` (o el gestor de tu distribución).

Configura tu identidad una sola vez:

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
```

### uv (gestiona Python y las dependencias)

- **macOS / Linux:**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Windows (PowerShell):**

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Cierra y vuelve a abrir la terminal y comprueba: `uv --version`.

No hace falta instalar Python por separado: `uv` descarga Python 3.12 la primera vez
(lo indica `.python-version`).

### Editor

VS Code con las extensiones **Python**, **Jupyter** y **Ruff** (recomendado), o
JupyterLab (`uv run jupyter lab`, ya incluido en las dependencias).

## 2. Preparar el proyecto

1. Descomprime/copia esta carpeta en una ruta **sin sincronización en la nube**
   (evita OneDrive/Google Drive/Dropbox: bloquean ficheros y ralentizan `.venv`).
   Mejor sin espacios ni tildes, p. ej. `~/mlops/proyecto`.
2. Abre una terminal **en esa carpeta** e instala:

   ```bash
   uv sync
   ```

   Crea `.venv/` con Python 3.12 y las versiones exactas del lock.

3. Inicializa tu repositorio Git y guarda el estado inicial:

   ```bash
   git init
   git add .
   git commit -m "Estado inicial del proyecto de prácticas"
   ```

## 3. Ejecutar el smoke test

```bash
uv run python scripts/smoke_test.py
```

(o `make smoke-test`). Comprueba, sin tocar tu proyecto, todos los puntos del
preflight. La primera ejecución tarda 1–2 minutos (se compilan cachés). Resultado
esperado:

```text
✅ Versión de Python (…) 3.12.x …
✅ Git disponible …
✅ Librerías del núcleo importan …
✅ Extras opcionales (no bloquean) …
✅ Ruff: lint y formato del proyecto …
✅ Datos incluidos … churn.csv (7043, 7), churn_nuevos.csv (50, 6)
✅ Entrenar, guardar y recargar un modelo (joblib) …
✅ MLflow: registrar un run en SQLite local …
✅ FastAPI + Uvicorn + requests: petición HTTP real … 200 … 422 …
✅ Streamlit: ejecutar una app de prueba …
✅ Evidently: informe de drift HTML …
✅ Notebook baseline: ejecución de arriba abajo …
✅ Puertos locales libres (5001, 8000, 8501) libres

✅ Entorno listo para las prácticas.
```

## 4. Checklist de preflight

El smoke test cubre casi todo; completa a mano lo que falta:

- [ ] `uv run python --version` muestra Python 3.12.x.
- [ ] `git --version` responde y has hecho el primer commit.
- [ ] `uv sync` terminó sin errores (existe la carpeta `.venv`).
- [ ] `notebooks/baseline.ipynb` abre en VS Code/JupyterLab y se ejecuta entero
      con *Restart & Run All* (kernel: `.venv`).
- [ ] `uv run ruff check .` y `uv run ruff format --check .` no muestran avisos.
- [ ] El smoke test termina con `✅ Entorno listo para las prácticas.`
- [ ] Abres <http://127.0.0.1:5001> tras `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`
      (se verá vacío; para con `Ctrl+C`).

Entrenar el modelo, levantar la API y la UI con el código del proyecto se hace en
clase: esas piezas tienen `TODO` y **es normal** que todavía fallen con
`NotImplementedError`.

Extras opcionales (solo si el docente lo pide, no bloquean el curso):

```bash
uv sync --group optuna      # demo de Optuna
uv sync --group onnx        # demo de ONNX
```

## 5. Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `uv: command not found` | Cierra y abre la terminal. Si persiste, añade `~/.local/bin` (macOS/Linux) o `%USERPROFILE%\.local\bin` (Windows) al `PATH`. |
| Windows: «la ejecución de scripts está deshabilitada» | Usa siempre `uv run …` (no actives el venv a mano) o ejecuta `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| `uv sync` falla por red/proxy corporativo | Prueba otra red; con proxy: `export HTTPS_PROXY=http://proxy:puerto` (PowerShell: `$env:HTTPS_PROXY=...`). Si no se resuelve, avisa: hay entorno de respaldo. |
| Smoke test: `Versión de Python … se espera (3, 12)` | Estás usando otro Python. Ejecuta con `uv run …` desde la carpeta del proyecto, o `uv python install 3.12` y `uv sync`. |
| Puertos ocupados | Cierra el proceso que los usa (otra API/Streamlit abierto) o cambia el puerto (`--port 8001`). Usamos 5001 para MLflow porque en macOS el 5000 lo ocupa AirPlay. |
| Smoke test: `Ruff …` con avisos | No has tocado nada y falla: avisa. Si has editado código: `uv run ruff format .` |
| El notebook no encuentra `../data/churn.csv` | Ábrelo desde la carpeta `notebooks/` del proyecto y con el kernel `.venv`. |
| Todo va muy lento / errores raros de ficheros | Mueve el proyecto fuera de carpetas sincronizadas en la nube y repite `uv sync`. |
| Mac con chip Apple: avisos de arquitectura | Asegúrate de que la terminal no corre con Rosetta y repite `uv sync`. |

Para ver la traza completa de un fallo del smoke test:
`SMOKE_DEBUG=1 uv run python scripts/smoke_test.py`.

**Si no consigues resolverlo:** comparte en el canal de dudas tu sistema operativo y
la salida completa del smoke test. No pierdas la sesión instalando: el docente
tiene un entorno de respaldo.
