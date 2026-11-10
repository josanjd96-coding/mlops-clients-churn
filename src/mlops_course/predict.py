"""Carga del modelo y predicción (en memoria y batch sobre CSV).

Uso batch (se crea en la sesión 1 y se usa en la sesión 3), desde la raíz del proyecto:

    uv run python -m mlops_course.predict --input data/churn_nuevos.csv \
        --output reports/predicciones.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlops_course.config import load_config, resolve_path, tracking_uri
from mlops_course.data import FEATURES


def load_model(cfg: dict) -> tuple[Any, dict]:
    """Carga el pipeline local y sus metadatos desde las rutas de la configuración.

    Solo cargar artefactos joblib generados por nosotros: deserializar un fichero
    no confiable puede ejecutar código.
    """
    # TODO S1: comprobar que existe cfg["artefactos"]["modelo"] (FileNotFoundError
    # con un mensaje útil si no), cargarlo con joblib.load y leer el JSON de
    # cfg["artefactos"]["metadatos"]. Devolver (modelo, metadatos).
    ruta_modelo = resolve_path(cfg["artefactos"]["metadatos"])
    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f'No exite {ruta_modelo}. Para ello tienes que entrenar antes un modelo'
        )

    # Deserializamos nuestro modelo

    raise NotImplementedError("TODO S1: load_model")


def load_registered_model(cfg: dict, version: str):
    """Carga una versión concreta del Model Registry de MLflow."""
    # TODO S2: set_tracking_uri(tracking_uri(cfg)) y devolver
    # mlflow.sklearn.load_model("models:/<modelo_registrado>/<version>").
    raise NotImplementedError("TODO S2: load_registered_model")


def predict_df(model, df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Devuelve `probabilidad` de abandono y clase `abandona` para cada fila."""
    # TODO S1: usar model.predict_proba sobre df[FEATURES], quedarse con la
    # columna de la clase 1 y aplicar el umbral. Devolver un DataFrame con las
    # columnas "probabilidad" y "abandona" y el mismo índice que df.
    y_proba = model.predict_proba(df[FEATURES])[:,1]
    y_pred = ()
    raise NotImplementedError("TODO S1: predict_df")


def batch_predict(cfg: dict, entrada: Path, salida: Path) -> pd.DataFrame:
    """Scoring batch: lee un CSV, predice y guarda el resultado con la versión."""
    # TODO S1: cargar modelo y metadatos (load_model), leer el CSV de entrada,
    # comprobar columnas, unir las predicciones (predict_df) al DataFrame original,
    # añadir columnas "umbral" y "model_version", guardar en `salida` y devolverlo.
    raise NotImplementedError("TODO S1: batch_predict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción batch sobre un CSV")
    parser.add_argument("--entorno", default=None)
    parser.add_argument("--input", default="data/churn_nuevos.csv")
    parser.add_argument("--output", default="reports/predicciones.csv")
    args = parser.parse_args()
    cfg = load_config(args.entorno)
    batch_predict(cfg, resolve_path(args.input), resolve_path(args.output))


if __name__ == "__main__":
    main()
