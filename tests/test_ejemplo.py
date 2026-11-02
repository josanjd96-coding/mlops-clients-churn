"""Tests de ejemplo (bonus, no obligatorios).

Ejecutar desde la raíz del proyecto:

    uv run --group dev pytest

Los tests que dependen de un TODO aún sin resolver se marcan como *skipped*
(no como fallo). A medida que completes las prácticas, pasarán a ejecutarse.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from mlops_course.config import load_config, resolve_path
from mlops_course.data import CATEGORIAS, FEATURES
from mlops_course.models import build_estimator, build_pipeline
from mlops_course.predict import predict_df
from mlops_course.schemas import Cliente, Prediccion

EJEMPLO = {
    "antiguedad_meses": 12,
    "cuota_mensual": 70.35,
    "tipo_contrato": "mensual",
    "tipo_internet": "fibra",
    "metodo_pago": "cheque_electronico",
}


def pendiente(funcion, *args):
    """Ejecuta `funcion`; si es un TODO sin resolver, salta el test."""
    try:
        return funcion(*args)
    except NotImplementedError as error:
        pytest.skip(str(error))


@pytest.fixture
def datos_pequenos() -> pd.DataFrame:
    filas = []
    for i in range(40):
        filas.append(
            {
                "antiguedad_meses": i,
                "cuota_mensual": 20.0 + i,
                "tipo_contrato": CATEGORIAS["tipo_contrato"][i % 3],
                "tipo_internet": CATEGORIAS["tipo_internet"][i % 3],
                "metodo_pago": CATEGORIAS["metodo_pago"][i % 4],
                "abandona": int(i < 20),
            }
        )
    return pd.DataFrame(filas)


def test_configuracion_dev_sin_rutas_absolutas():
    cfg = load_config("dev")
    assert cfg["entorno"] == "dev"
    assert not cfg["datos"]["ruta"].startswith("/")
    assert resolve_path(cfg["datos"]["ruta"]).exists()


def test_factory_rechaza_tipos_desconocidos():
    with pytest.raises(ValueError, match="no soportado"):
        build_estimator("modelo_inventado")


def test_pipeline_devuelve_probabilidad_y_clase(datos_pequenos):
    model = pendiente(build_pipeline, "regresion_logistica", {"max_iter": 500})
    model.fit(datos_pequenos[FEATURES], datos_pequenos["abandona"])
    resultado = pendiente(predict_df, model, datos_pequenos, 0.5)
    assert list(resultado.columns) == ["probabilidad", "abandona"]
    assert len(resultado) == len(datos_pequenos)
    assert resultado["probabilidad"].between(0, 1).all()
    assert set(resultado["abandona"]) <= {0, 1}


def test_pipeline_tolera_categorias_nuevas(datos_pequenos):
    model = pendiente(build_pipeline, "regresion_logistica", {"max_iter": 500})
    model.fit(datos_pequenos[FEATURES], datos_pequenos["abandona"])
    nueva = datos_pequenos.head(1).assign(metodo_pago="bizum")
    assert pendiente(predict_df, model, nueva, 0.5).shape == (1, 2)


def test_contrato_de_entrada_acepta_el_ejemplo():
    if "cuota_mensual" not in Cliente.model_fields:
        pytest.skip("TODO S3: Cliente incompleto")
    assert Cliente(**EJEMPLO).cuota_mensual == pytest.approx(70.35)


@pytest.mark.parametrize(
    "cambio",
    [
        {"cuota_mensual": -1},
        {"antiguedad_meses": 500},
        {"tipo_contrato": "semanal"},
        {"campo_extra": 1},
    ],
)
def test_contrato_de_entrada_rechaza_valores_invalidos(cambio):
    if "cuota_mensual" not in Cliente.model_fields:
        pytest.skip("TODO S3: Cliente incompleto")
    with pytest.raises(ValidationError):
        Cliente(**(EJEMPLO | cambio))


def test_contrato_de_salida():
    Prediccion(abandona=1, probabilidad=0.7, umbral=0.5, model_version="3")
    with pytest.raises(ValidationError):
        Prediccion(abandona=1, probabilidad=1.7, umbral=0.5, model_version="3")
