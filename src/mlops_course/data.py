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
    df = pd.read_csv(ruta)
    columnas_faltantes = [c for c in [*FEATURES, TARGET] if c not in df.columns]

    if columnas_faltantes:
        raise ValueError(f'Faltan columnas en el CSV cuya ruta es {ruta}: {columnas_faltantes}')

    return df

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

    df_resto, df_test  = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=semilla
    )

    df_train, df_val = train_test_split(
        df_resto, test_size=val_size, stratify=df_resto[TARGET], random_state=semilla
    )

    # print(f"train: {len(df_train)}, validación: {len(df_val)}, test: {len(df_test)}")
    return df_train, df_val, df_test

    raise NotImplementedError("TODO S1: split_data")


def x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features y target."""

    X = df[FEATURES]
    y = df[TARGET]
    
    return X, y
