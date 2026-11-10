"""Construcción del pipeline de entrenamiento.

`build_estimator` es una *factory simple*: una función que construye el
clasificador indicado en la configuración y falla de forma explícita si el tipo
no está soportado. No hace falta una jerarquía de clases para esto.
"""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlops_course.data import CATEGORIAS, CATEGORICAS, NUMERICAS

ESTIMADORES = {
    "regresion_logistica": LogisticRegression,
    "random_forest": RandomForestClassifier,
    'decision_tree' : DecisionTreeClassifier
}


def build_estimator(tipo: str, parametros: dict[str, Any] | None = None):
    """Factory simple: devuelve el clasificador sin entrenar."""
    if tipo not in ESTIMADORES:
        raise ValueError(
            f"Tipo de modelo no soportado: {tipo!r}. Usa {list(ESTIMADORES)}"
        )
    return ESTIMADORES[tipo](**(parametros or {}))


def build_pipeline(tipo: str, parametros: dict[str, Any] | None = None) -> Pipeline:
    """Preprocesamiento + clasificador en un único objeto.

    Al estar dentro del pipeline, escalado y one-hot se ajustan solo con los datos
    de `fit` (train), lo que evita fuga de información hacia validación/test.
    """
    # TODO S1: trasladar aquí el pipeline del notebook:
    #   - ColumnTransformer con StandardScaler para NUMERICAS y OneHotEncoder
    #     (categories=[CATEGORIAS[c] for c in CATEGORICAS], handle_unknown="ignore")
    #     para CATEGORICAS.
    #   - Pipeline con los pasos "preprocesado" y "clasificador"
    #     (usa build_estimator(tipo, parametros)).

    preprocesamiento = ColumnTransformer(
        [
            ("numericas", StandardScaler(), NUMERICAS),
            ("categoricas", OneHotEncoder(
                categories=[CATEGORIAS[c] for c in CATEGORICAS],
                handle_unknown="ignore"
                ), CATEGORICAS),
        ]
    )

    modelo = Pipeline(
    [
        ("preprocesamiento", preprocesamiento),
        ("clasificador", build_estimator(tipo, parametros)),
    ]
)


    return modelo
