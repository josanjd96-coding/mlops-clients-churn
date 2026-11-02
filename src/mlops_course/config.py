"""Carga de configuración no sensible desde `configs/<entorno>.yaml`.

- El entorno se elige con la variable `APP_ENV` (por defecto `dev`).
- Las rutas del YAML son relativas a la raíz del proyecto.
- Los secretos NO van en el YAML: se leen de variables de entorno.
"""

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(entorno: str | None = None) -> dict[str, Any]:
    """Devuelve la configuración del entorno indicado como diccionario."""
    entorno = entorno or os.environ.get("APP_ENV", "dev")
    ruta = PROJECT_ROOT / "configs" / f"{entorno}.yaml"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la configuración {ruta}")
    with ruta.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Las variables de entorno tienen prioridad sobre el YAML.
    if uri := os.environ.get("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = uri
    return cfg


def resolve_path(ruta: str | Path) -> Path:
    """Convierte una ruta relativa del YAML en ruta absoluta dentro del proyecto."""
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else PROJECT_ROOT / ruta


def tracking_uri(cfg: dict[str, Any]) -> str:
    """URI de MLflow; las rutas sqlite relativas se anclan a la raíz del proyecto."""
    uri = cfg["mlflow"]["tracking_uri"]
    prefijo = "sqlite:///"
    if uri.startswith(prefijo) and not Path(uri[len(prefijo) :]).is_absolute():
        return prefijo + (PROJECT_ROOT / uri[len(prefijo) :]).as_posix()
    return uri
