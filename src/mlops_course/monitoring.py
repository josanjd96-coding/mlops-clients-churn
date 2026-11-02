"""Monitorización: ventanas de datos con drift controlado, informe Evidently y
comparación del modelo actual frente a un candidato.

Uso (desde la raíz del proyecto, con el modelo ya entrenado):

    uv run python -m mlops_course.monitoring simular --escenario covariables
    uv run python -m mlops_course.monitoring informe
    uv run python -m mlops_course.monitoring comparar

Escenarios de `simular` (todos con semilla fija):
- estable:     muestra aleatoria de test -> no debería haber drift relevante.
- covariables: remuestreo sesgado hacia cuotas altas, contrato mensual y cheque
               electrónico. Cambia P(X) pero NO la relación real X -> Y.
- concepto:    mismas features que `estable`, pero cambia la etiqueta en un
               segmento (clientes bianuales que ahora abandonan). Cambia P(Y|X)
               sin que los histogramas de features lo muestren.
"""

import argparse

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from mlops_course.config import load_config, resolve_path
from mlops_course.data import (
    CATEGORICAS,
    FEATURES,
    NUMERICAS,
    TARGET,
    load_data,
    split_data,
    x_y,
)
from mlops_course.models import build_pipeline
from mlops_course.predict import load_model, predict_df
from mlops_course.train import evaluate

TAMANO_VENTANA = 1000
MARGEN_PROMOCION = 0.01  # mejora mínima de ROC AUC para promover un candidato


def _particiones(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_data(resolve_path(cfg["datos"]["ruta"]))
    return split_data(
        df, cfg["datos"]["test_size"], cfg["datos"]["val_size"], cfg["semilla"]
    )


def simular_ventana(test: pd.DataFrame, escenario: str, semilla: int) -> pd.DataFrame:
    """Genera una ventana "actual" a partir de test, con un cambio controlado."""
    # TODO S4: devolver TAMANO_VENTANA filas de `test` según el escenario:
    #   - "estable"/"concepto": test.sample(..., replace=True, random_state=semilla)
    #   - "covariables": muestreo con pesos que favorezcan cuota_mensual alta,
    #     contrato "mensual" y pago "cheque_electronico" (parámetro weights=...).
    #   - "concepto": además, poner TARGET=1 en ~40 % de los clientes "bianual"
    #     (usa rng.random) para cambiar P(Y|X) sin tocar las features.
    raise NotImplementedError("TODO S4: simular_ventana")


def con_predicciones(model, df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Añade la probabilidad predicha para poder vigilar también prediction drift."""
    return df.assign(probabilidad=predict_df(model, df, umbral)["probabilidad"])


def informe_drift(referencia: pd.DataFrame, actual: pd.DataFrame, salida) -> dict:
    """Genera el informe HTML de Evidently y devuelve un resumen por columna."""
    # TODO S4: crear un DataDefinition con columnas numéricas (NUMERICAS +
    # "probabilidad") y categóricas; convertir referencia y actual con
    # Dataset.from_pandas; ejecutar Report([DataDriftPreset()]).run(...);
    # guardar con snapshot.save_html(str(salida)) y devolver un dict
    # {columna: {"metodo", "valor", "umbral", "drift"}} a partir de
    # snapshot.dict()["metrics"] (métricas cuyo metric_name empieza por
    # "ValueDrift("; su "config" trae column, method y threshold).
    raise NotImplementedError("TODO S4: informe_drift")


def comparar_modelos(cfg: dict, train: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Actual vs candidato con etiquetas "maduras" de la ventana actual.

    La mitad de la ventana (etiquetas ya confirmadas) se usa para reentrenar el
    candidato junto con train; la otra mitad, para evaluar a ambos por igual.
    """
    # TODO S4: separar la ventana en dos mitades (maduras / evaluacion), entrenar
    # un candidato con build_pipeline sobre train + maduras, evaluar actual y
    # candidato con evaluate() sobre `evaluacion` y decidir: promover solo si la
    # ROC AUC mejora al menos MARGEN_PROMOCION y el F1 no empeora.
    raise NotImplementedError("TODO S4: comparar_modelos")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitorización y drift")
    parser.add_argument("accion", choices=["simular", "informe", "comparar"])
    parser.add_argument(
        "--escenario",
        default="covariables",
        choices=["estable", "covariables", "concepto"],
    )
    parser.add_argument("--entorno", default=None)
    args = parser.parse_args()

    cfg = load_config(args.entorno)
    mon = cfg["monitorizacion"]
    ruta_ref, ruta_act = resolve_path(mon["referencia"]), resolve_path(mon["actual"])
    train, _, test = _particiones(cfg)

    if args.accion == "simular":
        ruta_ref.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(ruta_ref, index=False)
        ventana = simular_ventana(test, args.escenario, cfg["semilla"])
        ventana.to_csv(ruta_act, index=False)
        print(f"Referencia (train): {len(train)} filas -> {ruta_ref}")
        print(f"Ventana actual ({args.escenario}): {len(ventana)} filas -> {ruta_act}")
        print(
            f"Tasa de abandono observada: ref {train[TARGET].mean():.3f}"
            f" | actual {ventana[TARGET].mean():.3f}"
        )

    elif args.accion == "informe":
        model, metadatos = load_model(cfg)
        umbral = cfg["modelo"]["umbral"]
        referencia = con_predicciones(model, pd.read_csv(ruta_ref), umbral)
        actual = con_predicciones(model, pd.read_csv(ruta_act), umbral)
        resumen = informe_drift(referencia, actual, resolve_path(mon["informe"]))
        print(f"Modelo {metadatos['model_version']} · drift por columna:")
        for columna, r in resumen.items():
            marca = "DRIFT" if r["drift"] else "ok"
            print(
                f"  {columna:<18} {marca:<5} {r['valor']:.4f}"
                f" (umbral {r['umbral']}, {r['metodo']})"
            )
        print(
            f"Media de probabilidad predicha: ref {referencia.probabilidad.mean():.3f}"
            f" | actual {actual.probabilidad.mean():.3f}"
        )

    else:
        r = comparar_modelos(cfg, train, pd.read_csv(ruta_act))
        print(
            f"Modelo actual (versión {r['version_actual']}) vs candidato, mismos datos:"
        )
        for nombre in ("roc_auc", "f1", "recall", "precision"):
            print(
                f"  {nombre}: actual {r['actual'][nombre]:.4f}"
                f" | candidato {r['candidato'][nombre]:.4f}"
            )
        print(f"Decisión propuesta: {r['decision']}")
        print(f"Mejora de ROC AUC del candidato: {r['mejora_roc_auc']:+.4f}")


if __name__ == "__main__":
    main()
