# Heart Disease — datos comunes del proyecto final

`heart_disease.csv` es la copia común **cleveland-binary-v1** para todas las
parejas. Está incluida en el repositorio: no requiere cuenta, API ni conexión
para utilizarla. La ruta se mantiene en `data/alt/` por compatibilidad con el
notebook; **no es una elección alternativa para la entrega final**.

## Fuente, licencia y transformaciones

- Fuente: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease),
  exclusivamente el archivo `processed.cleveland.data`.
- Autores: Andras Janosi, William Steinbrunn, Matthias Pfisterer y Robert Detrano.
- Cita: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
  *Heart Disease* [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C52P4X>.
- Licencia de los datos: [Creative Commons Attribution 4.0 International
  (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
  Permite compartir y adaptar con atribución, enlace a la licencia e indicación
  de los cambios. Conservar esta documentación al redistribuir el CSV.
- Adaptación docente: cabecera añadida, números normalizados, `?` convertido en
  campo vacío y diagnóstico original `num` transformado a `target` binario
  (`0 → 0`; `1, 2, 3, 4 → 1`). Se conservan las 303 filas y su orden.
  No se imputan ausentes, se escalan variables ni se eliminan filas.

Hay **164 casos de clase 0 y 139 de clase 1**. Faltan 4 valores en `ca` y 2 en
`thal`. Los campos vacíos se leen como `NaN` con pandas; la imputación debe
aprenderse solo con entrenamiento.

**Atención:** este CSV sustituye una versión anterior del material con códigos
diferentes. En esta copia `cp` usa 1–4, `slope` 1–3, `thal` 3/6/7, y la clase
positiva representa presencia de enfermedad. No mezclar versiones.

## Diccionario de variables

| Columna | Tipo / códigos | Significado |
|---|---|---|
| `age` | Entero | Edad en años |
| `sex` | Categórica: 0, 1 | Sexo codificado en el original: 0 femenino, 1 masculino |
| `cp` | Categórica: 1–4 | 1 angina típica, 2 atípica, 3 dolor no anginoso, 4 asintomático |
| `trestbps` | Entero | Presión arterial en reposo, mmHg |
| `chol` | Entero | Colesterol sérico, mg/dl |
| `fbs` | Categórica: 0, 1 | Glucemia en ayunas > 120 mg/dl: 1 sí, 0 no |
| `restecg` | Categórica: 0–2 | 0 normal, 1 anomalía ST-T, 2 hipertrofia ventricular izquierda probable/definida |
| `thalach` | Entero | Frecuencia cardíaca máxima alcanzada |
| `exang` | Categórica: 0, 1 | Angina inducida por ejercicio: 1 sí, 0 no |
| `oldpeak` | Decimal | Depresión ST inducida por ejercicio respecto al reposo |
| `slope` | Categórica: 1–3 | Pendiente ST: 1 ascendente, 2 plana, 3 descendente |
| `ca` | Discreta: 0–3; admite ausentes | Número de vasos principales coloreados por fluoroscopia |
| `thal` | Categórica: 3, 6, 7; admite ausentes | 3 normal, 6 defecto fijo, 7 defecto reversible |
| `target` | Binaria: 0, 1 | **Objetivo**: 0 ausencia, 1 presencia según la etiqueta del dataset |

`target` nunca debe incluirse entre las variables predictoras, en el CSV de
inferencia ni en el cuerpo de una petición a la API. No contiene identificadores.
Las categorías numéricas no deben interpretarse automáticamente como continuas.

## Uso para alumnado y docente

Desde la raíz de `alumnos/` o de cualquier `docente/sesion-N/`:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/alt/heart_disease.csv")
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

La partición común es **80% entrenamiento / 20% test, estratificada y semilla
42** (242/61 filas). No ajustar transformaciones, configuraciones ni umbrales
con test. Para elegir entre ejecuciones, usar validación interna de entrenamiento.
El notebook `notebooks/alt_heart_disease.ipynb` es una ayuda inicial, no la solución
completa del proyecto.

`heart_disease.metadata.json` contiene el hash SHA-256 del original y del CSV,
columnas, clases y ausentes. Para comprobar la copia sin instalar dependencias:

```bash
python -c "import hashlib,json,pathlib; p=pathlib.Path('data/alt'); m=json.loads((p/'heart_disease.metadata.json').read_text()); h=hashlib.sha256((p/'heart_disease.csv').read_bytes()).hexdigest(); print(h); assert h == m['csv_sha256'], 'CSV modificado'"
```

## Mantenimiento docente

El original descargado y su fecha están en `docente/datos/raw/`. Desde la raíz
del repositorio completo:

```bash
# Regenerar y sincronizar las copias usando el original incluido (sin red):
python docente/datos/preparar_heart.py

# Solo si se necesita recuperar el original desde UCI:
python docente/datos/preparar_heart.py --descargar
```

La fuente de distribución es
`docente/_mantenimiento/plantilla/data/alt/`; el script sincroniza CSV, metadatos
y este README con alumnado y los cuatro checkpoints. Las regeneraciones generales
del curso también los conservan. No editar los CSV a mano ni usar
`docente/datos/raw/heart_disease.csv`, que pertenece al material antiguo.

**Uso académico exclusivamente.** Son datos clínicos históricos, no datos
sintéticos. Su disponibilidad pública no elimina los riesgos ni permite validar
un sistema diagnóstico. No añadir datos personales, intentar reidentificar
registros ni utilizar las predicciones para decisiones clínicas.
