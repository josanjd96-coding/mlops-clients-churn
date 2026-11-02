"""Smoke test del entorno: comprueba el preflight sin depender de los TODO.

Ejecutar desde la raíz del proyecto:

    uv run python scripts/smoke_test.py            # comprobación completa
    uv run python scripts/smoke_test.py --rapido   # sin ejecutar el notebook

Todo se hace en una carpeta temporal: no se modifica tu proyecto ni se usa red
externa. Si algo falla, copia la salida completa y compártela en el canal de dudas.
"""

import argparse
import importlib
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PYTHON_ESPERADO = (3, 12)
PUERTOS = {5001: "MLflow UI", 8000: "FastAPI", 8501: "Streamlit"}
LIBRERIAS = [
    "pandas",
    "numpy",
    "sklearn",
    "joblib",
    "yaml",
    "mlflow",
    "fastapi",
    "uvicorn",
    "pydantic",
    "httpx",
    "streamlit",
    "requests",
    "evidently",
    "jupyterlab",
    "nbconvert",
]
OPCIONALES = ["pytest", "optuna", "skl2onnx", "onnxruntime"]

resultados: list[tuple[str, bool, str]] = []


def paso(nombre: str):
    """Decorador: ejecuta la comprobación y registra ✅/❌ sin detener el resto."""

    def envoltorio(funcion):
        def ejecutar(*args):
            inicio = time.perf_counter()
            try:
                detalle = funcion(*args) or ""
                ok = True
            except Exception as error:  # se informa de cualquier fallo
                detalle = f"{type(error).__name__}: {error}"
                if os.environ.get("SMOKE_DEBUG"):
                    traceback.print_exc()
                ok = False
            segundos = time.perf_counter() - inicio
            icono = "✅" if ok else "❌"
            print(f"{icono} {nombre} ({segundos:.1f}s) {detalle}", flush=True)
            resultados.append((nombre, ok, detalle))

        return ejecutar

    return envoltorio


@paso("Versión de Python")
def comprobar_python():
    actual = sys.version_info[:2]
    if actual != PYTHON_ESPERADO:
        raise RuntimeError(f"tienes {actual}, se espera {PYTHON_ESPERADO}")
    return f"{sys.version.split()[0]} en {sys.prefix}"


@paso("Git disponible")
def comprobar_git():
    salida = subprocess.run(
        ["git", "--version"], capture_output=True, text=True, check=True
    )
    return salida.stdout.strip()


@paso("Librerías del núcleo importan")
def comprobar_imports():
    versiones = []
    for nombre in LIBRERIAS:
        modulo = importlib.import_module(nombre)
        versiones.append(f"{nombre} {getattr(modulo, '__version__', '?')}")
    return "\n     " + "\n     ".join(versiones)


@paso("Extras opcionales (no bloquean)")
def comprobar_opcionales():
    estado = []
    for nombre in OPCIONALES:
        try:
            importlib.import_module(nombre)
            estado.append(f"{nombre} ok")
        except ImportError:
            estado.append(f"{nombre} no instalado")
    return "· ".join(estado)


@paso("Ruff: lint y formato del proyecto")
def comprobar_ruff():
    for orden in (["check", "."], ["format", "--check", "."]):
        salida = subprocess.run(
            [sys.executable, "-m", "ruff", *orden],
            cwd=RAIZ,
            capture_output=True,
            text=True,
        )
        if salida.returncode != 0:
            raise RuntimeError(f"ruff {' '.join(orden)}:\n{salida.stdout}")
    return "ruff check . y ruff format --check . sin avisos"


@paso("Datos incluidos")
def comprobar_datos():
    import pandas as pd

    df = pd.read_csv(RAIZ / "data" / "churn.csv")
    nuevos = pd.read_csv(RAIZ / "data" / "churn_nuevos.csv")
    esperadas = {
        "antiguedad_meses",
        "cuota_mensual",
        "tipo_contrato",
        "tipo_internet",
        "metodo_pago",
        "abandona",
    }
    if faltan := esperadas - set(df.columns):
        raise RuntimeError(f"faltan columnas: {faltan}")
    return f"churn.csv {df.shape}, churn_nuevos.csv {nuevos.shape}"


def entrenar_mini():
    """Pipeline pequeño, independiente del código de las prácticas."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(RAIZ / "data" / "churn.csv").sample(1500, random_state=0)
    X, y = df.drop(columns=["id_cliente", "abandona"]), df["abandona"]
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), ["antiguedad_meses", "cuota_mensual"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["tipo_contrato", "tipo_internet", "metodo_pago"],
            ),
        ]
    )
    model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=500))])
    return model.fit(X, y), X


@paso("Entrenar, guardar y recargar un modelo (joblib)")
def comprobar_modelo(tmp: Path):
    import joblib
    import numpy as np

    model, X = entrenar_mini()
    ruta = tmp / "artifacts" / "model.joblib"
    ruta.parent.mkdir(parents=True)
    joblib.dump(model, ruta)
    recargado = joblib.load(ruta)
    if not np.allclose(model.predict_proba(X), recargado.predict_proba(X)):
        raise RuntimeError("el modelo recargado no predice igual")
    return f"artefacto de prueba {ruta.stat().st_size // 1024} KB"


@paso("MLflow: registrar un run en SQLite local")
def comprobar_mlflow(tmp: Path):
    import mlflow

    logging.getLogger("mlflow").setLevel(logging.ERROR)
    model, X = entrenar_mini()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("smoke-test")
    with mlflow.start_run() as run:
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("auc", 0.8)
        mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle"
        )
    recuperado = mlflow.get_run(run.info.run_id)
    cargado = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    cargado.predict(X.head(2))
    return f"run {run.info.run_id[:8]} con métrica auc={recuperado.data.metrics['auc']}"


def puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@paso("FastAPI + Uvicorn + requests: petición HTTP real")
def comprobar_api():
    import requests
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    class Entrada(BaseModel):
        cuota_mensual: float = Field(gt=0)

    app = FastAPI()

    @app.post("/predict")
    def predict(entrada: Entrada) -> dict:
        return {"probabilidad": 0.5, "cuota": entrada.cuota_mensual}

    puerto = puerto_libre()
    config = uvicorn.Config(app, host="127.0.0.1", port=puerto, log_level="error")
    servidor = uvicorn.Server(config)
    hilo = threading.Thread(target=servidor.run, daemon=True)
    hilo.start()
    try:
        for _ in range(50):
            if servidor.started:
                break
            time.sleep(0.1)
        url = f"http://127.0.0.1:{puerto}/predict"
        ok = requests.post(url, json={"cuota_mensual": 10}, timeout=5)
        ko = requests.post(url, json={"cuota_mensual": -1}, timeout=5)
        if ok.status_code != 200 or ko.status_code != 422:
            raise RuntimeError(f"códigos {ok.status_code}/{ko.status_code}")
    finally:
        servidor.should_exit = True
        hilo.join(timeout=5)
    return "200 con entrada válida y 422 con entrada inválida"


@paso("Streamlit: ejecutar una app de prueba")
def comprobar_streamlit(tmp: Path):
    from streamlit.testing.v1 import AppTest

    contexto = "streamlit.runtime.scriptrunner_utils.script_run_context"
    for nombre in ("streamlit", contexto):
        logging.getLogger(nombre).setLevel(logging.ERROR)

    script = tmp / "app_prueba.py"
    script.write_text(
        "import streamlit as st\n"
        "with st.form('f'):\n"
        "    x = st.number_input('x', value=1)\n"
        "    enviado = st.form_submit_button('Predecir')\n"
        "if enviado:\n"
        "    st.metric('Probabilidad', f'{x * 0.5:.0%}')\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=30).run()
    app.button[0].click().run()
    if app.exception or not app.metric:
        raise RuntimeError("la app de prueba no mostró la métrica")
    return f"formulario enviado, métrica {app.metric[0].value}"


@paso("Evidently: informe de drift HTML")
def comprobar_evidently(tmp: Path):
    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset

    df = pd.read_csv(RAIZ / "data" / "churn.csv").drop(columns=["id_cliente"])
    referencia, actual = df.iloc[:2000], df.iloc[2000:3000].copy()
    actual["cuota_mensual"] *= 1.3
    snapshot = Report([DataDriftPreset()]).run(
        current_data=actual, reference_data=referencia
    )
    salida = tmp / "drift.html"
    snapshot.save_html(str(salida))
    return f"informe de {salida.stat().st_size // 1024} KB generado"


@paso("Notebook baseline: ejecución de arriba abajo")
def comprobar_notebook(tmp: Path):
    notebook = RAIZ / "notebooks" / "baseline.ipynb"
    salida = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output-dir",
            str(tmp),
            str(notebook),
        ],
        cwd=RAIZ / "notebooks",
        capture_output=True,
        text=True,
    )
    if salida.returncode != 0:
        raise RuntimeError(salida.stderr[-800:])
    return "todas las celdas ejecutadas sin error"


@paso("Puertos locales libres (5001, 8000, 8501)")
def comprobar_puertos():
    ocupados = []
    for puerto, servicio in PUERTOS.items():
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", puerto)) == 0:
                ocupados.append(f"{puerto} ({servicio})")
    if ocupados:
        raise RuntimeError(
            f"ocupados: {', '.join(ocupados)}. Cierra esos procesos o usa otro puerto."
        )
    return "libres"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rapido", action="store_true", help="omite el notebook")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

    print(f"Smoke test del proyecto en {RAIZ}\n", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="mlops-smoke-"))
    try:
        comprobar_python()
        comprobar_git()
        comprobar_imports()
        comprobar_opcionales()
        comprobar_ruff()
        comprobar_datos()
        comprobar_modelo(tmp)
        comprobar_mlflow(tmp)
        comprobar_api()
        comprobar_streamlit(tmp)
        comprobar_evidently(tmp)
        if not args.rapido:
            comprobar_notebook(tmp)
        comprobar_puertos()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    fallos = [nombre for nombre, ok, _ in resultados if not ok]
    print()
    if fallos:
        print(f"❌ {len(fallos)} comprobación(es) fallida(s): {', '.join(fallos)}")
        print("Revisa SETUP.md (sección 'Problemas frecuentes').")
        return 1
    print("✅ Entorno listo para las prácticas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
