# Datos del proyecto

Todos los datos son **locales**: no hace falta descargar nada durante las sesiones.

## `churn.csv` — dataset principal

- **Origen:** *IBM Telco Customer Churn* (dataset de ejemplo de IBM, datos ficticios).
- **Licencia:** Apache License 2.0.
- **Datos personales:** no contiene; los clientes son sintéticos.
- **Tamaño:** 7.043 filas, tasa de abandono ≈ 26,5 %.
- **Transformación aplicada:** se seleccionaron 5 variables, se tradujeron
  nombres y categorías al español y se convirtió el target a 0/1.

| Columna | Tipo | Valores | Descripción |
|---|---|---|---|
| `id_cliente` | texto | p. ej. `7590-VHVEG` | Identificador. **No** es una feature. |
| `antiguedad_meses` | entero | 0–72 | Meses que lleva como cliente. |
| `cuota_mensual` | decimal | 18,25–118,75 | Importe mensual en euros. |
| `tipo_contrato` | categórica | `mensual`, `anual`, `bianual` | Duración del contrato. |
| `tipo_internet` | categórica | `dsl`, `fibra`, `sin_internet` | Servicio de internet contratado. |
| `metodo_pago` | categórica | `cheque_electronico`, `cheque_postal`, `transferencia`, `tarjeta` | Forma de pago. |
| `abandona` | 0/1 | **target** | 1 si el cliente se dio de baja. |

Ejemplo de entrada para el modelo (contrato de la API en la práctica 3):

```json
{
  "antiguedad_meses": 12,
  "cuota_mensual": 70.35,
  "tipo_contrato": "mensual",
  "tipo_internet": "fibra",
  "metodo_pago": "cheque_electronico"
}
```

## `churn_nuevos.csv` — scoring batch

50 clientes **sin la columna `abandona`**, como llegarían para predecir. Se han
extraído del propio dataset con semilla fija, así que es un ejemplo didáctico, no
una evaluación del modelo.

## Ficheros que se generan en la práctica 4 (ignorados por Git)

- `churn_referencia.csv`: datos de referencia (partición de train).
- `churn_actual.csv`: ventana "actual" con un cambio sintético controlado.

Se regeneran con `uv run python -m mlops_course.monitoring simular`.

## `alt/heart_disease.csv` — proyecto final común

Dataset oficial UCI Cleveland (303 filas, 13 variables predictoras y `target`
binario: 1 = presencia de enfermedad). Es el dataset obligatorio para todas las
parejas en el [proyecto final](../proyecto-final.md); las prácticas mantienen Churn.

Incluye ausentes en `ca` y `thal`, que deben imputarse dentro del pipeline.
Consultar [diccionario, licencia CC BY 4.0 y carga](alt/README.md), y
`alt/heart_disease.metadata.json` para verificar su versión y hash. El notebook
`notebooks/alt_heart_disease.ipynb` ayuda a iniciar el trabajo.
