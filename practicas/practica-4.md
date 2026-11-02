# Práctica 4 — Drift, monitorización y cierre del proyecto

**Objetivo:** comparar datos de referencia con una ventana actual, interpretar la
señal **sin confundir drift con degradación** y proponer una respuesta razonada.

**Duración:** 150 min · **Requisito:** modelo entrenado (práctica 2).

## Qué debe quedar hecho

- [ ] `reports/drift_report.html` generado con Evidently.
- [ ] Ejecución de los tres escenarios y comparación actual vs candidato.
- [ ] Reflexión breve (sección "Monitorización" de tu README) con referencia,
      ventana, señal observada, limitaciones y siguiente acción.

## Conceptos (0–15 min)

| Tipo | Qué cambia | ¿Se ve sin etiquetas? |
|---|---|---|
| Data drift (covariables) | P(X) | Sí |
| Prediction drift | distribución de las predicciones | Sí |
| Label drift | P(Y) | Solo con etiquetas |
| Concept drift | P(Y\|X) | Solo con etiquetas (rendimiento) |

## Paso 1 — Referencia y ventana actual (15–65 min)

Completa `monitoring.py` → `simular_ventana` (`TODO S4`) y genera los datos:

```bash
uv run python -m mlops_course.monitoring simular --escenario covariables
```

- **Referencia:** la partición de train (`data/churn_referencia.csv`).
- **Ventana actual:** 1.000 filas a partir de test con un cambio controlado
  (`data/churn_actual.csv`).

Escenarios: `estable` (sin cambio), `covariables` (más cuotas altas, contrato
mensual y cheque electrónico) y `concepto` (mismas X, cambia la etiqueta en un
segmento).

Antes de usar Evidently, inspecciona con pandas: tipos, nulos, `describe()` y
`value_counts(normalize=True)` de ambas tablas.

## Paso 2 — Informe Evidently (75–105 min)

Completa `monitoring.py` → `informe_drift` (`TODO S4`):

```bash
uv run python -m mlops_course.monitoring informe
```

Abre `reports/drift_report.html` en el navegador. Resultado orientativo para
`covariables`:

```text
Modelo 2 · drift por columna:
  antiguedad_meses   DRIFT 0.14… (umbral 0.1, Wasserstein distance (normed))
  cuota_mensual      DRIFT 0.65… …
  probabilidad       DRIFT 0.66… …
  tipo_contrato      DRIFT …     (umbral 0.1, Jensen-Shannon distance)
  …
```

Repite con `--escenario estable` y `--escenario concepto`. **¿Qué detecta el informe
en `concepto`? ¿Por qué?**

## Paso 3 — Actual vs candidato (105–140 min)

Completa `monitoring.py` → `comparar_modelos` (`TODO S4`). Simula que llegan las
etiquetas de la mitad de la ventana: el candidato se entrena con train + esa mitad
y **ambos** se evalúan con la otra mitad.

```bash
uv run python -m mlops_course.monitoring comparar
```

Criterio didáctico de promoción: el candidato mejora ROC AUC al menos 0,01 y no
empeora F1. Completa la tabla:

| Escenario | ¿Drift en X? | ROC AUC actual | ROC AUC candidato | Decisión |
|---|---|---|---|---|
| estable | | | | |
| covariables | | | | |
| concepto | | | | |

Preguntas:

1. En `covariables` hay drift claro, ¿el modelo ha empeorado? ¿Reentrenar mejora?
2. En `concepto` apenas hay drift en X, ¿por qué cae el rendimiento?
3. ¿Qué harías sin etiquetas? (investigar, esperar etiquetas, recalibrar…)
4. ¿Qué limitaciones tiene una ventana de 1.000 filas y un umbral fijo de 0,1?

## Cierre y entrega (140–150 min)

Revisa la checklist de entrega que te dará el docente. Mínimo:

1. Repositorio con código, notebook y README (instalación, ejecución, ejemplo).
2. Configuración sin secretos y dependencias fijadas (`uv.lock`).
3. Evidencia de experimentos (captura de MLflow) y de predicciones.
4. Captura de la UI consumiendo la API.
5. Informe/captura de drift y reflexión.
6. Demo de 5–8 minutos.

`git commit -m "S4: drift, comparación y reflexión"`.
