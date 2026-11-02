"""Entrenamiento del modelo de abandono.

Uso (desde la raíz del proyecto):

    uv run python -m mlops_course.train               # baseline
    uv run python -m mlops_course.train --busqueda    # sesión 2: GridSearchCV acotado
    APP_ENV=preprod uv run python -m mlops_course.train
"""

import argparse
import json
from datetime import datetime
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from mlops_course.config import load_config, resolve_path, tracking_uri
from mlops_course.data import load_data, split_data, x_y
from mlops_course.models import build_pipeline


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, umbral: float) -> dict:
    """Métricas de clasificación con el umbral de decisión configurado."""
    # TODO S1: calcular la probabilidad de la clase 1 con predict_proba, aplicar el
    # umbral y devolver un dict con roc_auc, f1, precision, recall y accuracy.
    raise NotImplementedError("TODO S1: evaluate")


def save_artifacts(model: Pipeline, metadatos: dict[str, Any], cfg: dict) -> None:
    """Guarda el pipeline (joblib) y sus metadatos (JSON) en `artifacts/`."""
    # TODO S1: crear la carpeta si no existe, guardar el modelo con joblib.dump en
    # cfg["artefactos"]["modelo"] y los metadatos con json.dump en
    # cfg["artefactos"]["metadatos"]. Usa resolve_path para las rutas.
    raise NotImplementedError("TODO S1: save_artifacts")


def search(cfg: dict, train: pd.DataFrame) -> tuple[Pipeline, dict, float]:
    """Búsqueda acotada de hiperparámetros con validación cruzada SOLO sobre train."""
    # TODO S2: crear un GridSearchCV con build_pipeline(...), param_grid,
    # cv=StratifiedKFold(..., shuffle=True, random_state=semilla) y scoring
    # tomados de cfg["busqueda"]. Ajustarlo SOLO con train y devolver
    # (best_estimator_, best_params_, best_score_).
    raise NotImplementedError("TODO S2: search")


def log_to_mlflow(
    cfg: dict,
    model: Pipeline,
    params: dict,
    metricas: dict[str, float],
    X_ejemplo: pd.DataFrame,
    run_name: str,
) -> tuple[str, str]:
    """Registra parámetros, métricas y modelo. Devuelve (versión registrada, run_id)."""
    # TODO S2: set_tracking_uri(tracking_uri(cfg)), set_experiment, y dentro de
    # mlflow.start_run(run_name=...): log_params, log_param("umbral", ...),
    # log_metrics, log_dict(cfg, "config.yaml") y mlflow.sklearn.log_model(...,
    # name="model", input_example=X_ejemplo, registered_model_name=...,
    # serialization_format="cloudpickle"). Devolver (versión registrada, run_id).
    raise NotImplementedError("TODO S2: log_to_mlflow")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de abandono")
    parser.add_argument("--entorno", default=None, help="dev | preprod (o APP_ENV)")
    parser.add_argument("--busqueda", action="store_true", help="GridSearchCV acotado")
    args = parser.parse_args()

    cfg = load_config(args.entorno)
    df = load_data(resolve_path(cfg["datos"]["ruta"]))
    train, val, test = split_data(
        df, cfg["datos"]["test_size"], cfg["datos"]["val_size"], cfg["semilla"]
    )
    print(f"Filas -> train: {len(train)}, validación: {len(val)}, test: {len(test)}")
    X_train, y_train = x_y(train)
    umbral = cfg["modelo"]["umbral"]

    if args.busqueda:
        model, params, cv_score = search(cfg, train)
        run_name = "busqueda-grid"
    else:
        model = build_pipeline(cfg["modelo"]["tipo"], cfg["modelo"]["parametros"])
        model.fit(X_train, y_train)
        params = cfg["modelo"]["parametros"]
        run_name = "baseline"
        cv_score = None

    metricas = {f"val_{k}": v for k, v in evaluate(model, *x_y(val), umbral).items()}
    # El test se mira una vez, con el modelo ya elegido; no se usa para seleccionar.
    metricas |= {f"test_{k}": v for k, v in evaluate(model, *x_y(test), umbral).items()}
    if cv_score is not None:
        metricas["cv_mejor_score"] = cv_score
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor:.4f}")

    # TODO S2: sustituir esta línea por la llamada a log_to_mlflow(...)
    version, run_id = datetime.now().strftime("local-%Y%m%d-%H%M%S"), None

    metadatos = {
        "model_version": version,
        "mlflow_run_id": run_id,
        "run_name": run_name,
        "tipo": cfg["modelo"]["tipo"],
        "parametros": params,
        "umbral": umbral,
        "entorno": cfg["entorno"],
        "entrenado_en": datetime.now().isoformat(timespec="seconds"),
        "metricas": metricas,
    }
    save_artifacts(model, metadatos, cfg)

    # TODO S2: recargar el artefacto con joblib.load y comprobar con np.allclose
    # que predict_proba sobre validación coincide con el modelo en memoria.


if __name__ == "__main__":
    main()
