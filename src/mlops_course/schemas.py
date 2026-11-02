"""Contrato de entrada/salida del servicio de predicción.

Lo comparten la API (valida la entrada y la respuesta) y la interfaz Streamlit
(valida la respuesta recibida). Solo depende de Pydantic: la UI no necesita
importar scikit-learn ni cargar el modelo.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Cliente(BaseModel):
    """Una fila de entrada: las cinco features del modelo."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "antiguedad_meses": 12,
                    "cuota_mensual": 70.35,
                    "tipo_contrato": "mensual",
                    "tipo_internet": "fibra",
                    "metodo_pago": "cheque_electronico",
                }
            ]
        },
    )

    antiguedad_meses: int = Field(ge=0, le=120, description="Meses como cliente")
    # TODO S3: añadir aquí los otros cuatro campos con sus restricciones:
    #   cuota_mensual: float > 0 y <= 500, sin inf/nan (allow_inf_nan=False)
    #   tipo_contrato: Literal["mensual", "anual", "bianual"]
    #   tipo_internet: Literal["dsl", "fibra", "sin_internet"]
    #   metodo_pago: Literal["cheque_electronico", "cheque_postal",
    #                        "transferencia", "tarjeta"]


class Prediccion(BaseModel):
    """Respuesta de POST /predict."""

    abandona: Literal[0, 1]
    probabilidad: float = Field(ge=0, le=1, allow_inf_nan=False)
    umbral: float = Field(ge=0, le=1, allow_inf_nan=False)
    model_version: str = Field(min_length=1)
