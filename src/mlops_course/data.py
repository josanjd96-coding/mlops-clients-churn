"""Carga, validación y partición de datos del caso de abandono (churn)."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

NUMERICAS = ["antiguedad_meses", "cuota_mensual"]
CATEGORICAS = ["tipo_contrato", "tipo_internet", "metodo_pago"]
FEATURES = NUMERICAS + CATEGORICAS
TARGET = "abandona"

CATEGORIAS = {
    "tipo_contrato": ["mensual", "anual", "bianual"],
    "tipo_internet": ["dsl", "fibra", "sin_internet"],
    "metodo_pago": ["cheque_electronico", "cheque_postal", "transferencia", "tarjeta"],
}


def load_data(ruta: str | Path) -> pd.DataFrame:
    """Lee el CSV y comprueba que contiene las columnas esperadas."""
    # TODO S1: leer el CSV con pandas y lanzar ValueError si falta alguna columna
    # de FEATURES o TARGET. Pista: copia la celda de carga del notebook baseline.
    raise NotImplementedError("TODO S1: load_data")


def split_data(
    df: pd.DataFrame, test_size: float, val_size: float, semilla: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición reproducible y estratificada en train / validación / test.

    `val_size` es la fracción de lo que queda tras separar test.
    """
    # TODO S1: usar train_test_split dos veces (primero test, luego validación),
    # estratificando por TARGET y fijando random_state=semilla.
    raise NotImplementedError("TODO S1: split_data")


def x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features y target."""
    return df[FEATURES], df[TARGET]
