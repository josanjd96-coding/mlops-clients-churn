"""API de predicción con FastAPI.

Arranque (desde la raíz del proyecto):

    uv run uvicorn mlops_course.api:app --host 127.0.0.1 --port 8000

Documentación interactiva: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from mlops_course.config import load_config
from mlops_course.predict import load_model, predict_df
from mlops_course.schemas import Cliente, Prediccion

estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo UNA vez al arrancar, no en cada petición."""
    # TODO S3: cargar la configuración y el modelo (load_model) y guardar en
    # `estado` el modelo, el umbral y la model_version de los metadatos.
    yield
    estado.clear()


app = FastAPI(title="API de abandono de clientes", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Comprobación de vida: indica si hay modelo cargado y su versión."""
    return {"status": "ok", "model_version": estado.get("model_version")}


@app.post("/predict", response_model=Prediccion)
def predict(cliente: Cliente) -> Prediccion:
    """Predice la probabilidad de abandono de un cliente.

    Si la entrada no cumple el contrato, FastAPI responde 422 automáticamente.
    """
    # TODO S3: devolver 503 si no hay modelo; convertir `cliente` en un DataFrame
    # de una fila (cliente.model_dump()), usar predict_df con el umbral de
    # `estado` y devolver un objeto Prediccion con abandona, probabilidad,
    # umbral y model_version.
    raise HTTPException(status_code=501, detail="TODO S3: endpoint /predict")
