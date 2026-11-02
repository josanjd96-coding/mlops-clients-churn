# Máster en Inteligencia Artificial, Cloud Computing y DevOps

## MLOps

### Guion de entrega final — Borrador

**Profesor:** Sergio Benito Martín  
**Modalidad:** trabajo por parejas  
**Caso de uso común:** clasificación de presencia de enfermedad cardíaca con Heart Disease

> Pendiente de confirmar: fecha de entrega e inscripción de parejas.

## Índice

- Introducción
- Objetivo general
- Requisitos mínimos
- Criterios de evaluación
- Entregables obligatorios
- Conclusión
- Anexo: criterios de evaluación
- FAQs: preguntas y respuestas

## Introducción

Un modelo de Machine Learning no termina cuando se entrena. Para utilizarlo, es necesario poder reproducir su construcción, cargarlo desde otro proceso, realizar predicciones y analizar cambios en los datos que recibe.

En esta práctica se construirá un proyecto MLOps de alcance reducido que cubra ese recorrido. **Todas las parejas utilizarán el mismo caso de uso y la misma copia del dataset**, facilitada por el profesor.

El proyecto es exclusivamente académico: no constituye una herramienta de diagnóstico ni debe utilizarse para tomar decisiones clínicas.

## Objetivo general

Construir una solución reproducible que permita entrenar y guardar un pipeline, registrar los experimentos, ejecutar predicciones por lotes y mediante HTTP, y elaborar un informe sencillo de drift.

El objetivo no es obtener la mejor métrica de la clase ni incorporar muchas herramientas, sino demostrar un flujo completo, comprensible y ejecutable por otra persona.

### Caso de uso y datos comunes

Se utilizará el subconjunto **Cleveland de Heart Disease**, disponible en el [repositorio UCI Machine Learning](https://archive.ics.uci.edu/dataset/45/heart+disease). El CSV común está incluido en `data/alt/heart_disease.csv` dentro del proyecto distribuido al alumnado y de cada checkpoint docente. En el repositorio completo se encuentra en `alumnos/data/alt/heart_disease.csv`. No requiere descargas ni cuentas adicionales.

Su README adjunto incluye fuente, licencia **CC BY 4.0**, diccionario, transformaciones y ejemplo de carga; `heart_disease.metadata.json` permite verificar el hash de la copia **cleveland-binary-v1**. Todas las parejas utilizarán esta versión, no otra de Kaggle o del material antiguo.

La columna `target` ya contiene la transformación de la etiqueta original a clasificación binaria:

- **0:** ausencia de enfermedad según la etiqueta original, cuyo valor es 0.
- **1:** presencia de enfermedad según la etiqueta original, cuyo valor es superior a 0.

Se usarán las 13 variables predictoras habituales del subconjunto: **303 filas, 164 de clase 0 y 139 de clase 1**. Hay 4 ausentes en `ca` y 2 en `thal`, representados como campos vacíos, que deberán tratarse dentro del pipeline. No incluir `target` entre las variables predictoras ni en las entradas de inferencia.

Cada pareja creará una partición estratificada de **80% entrenamiento y 20% test, con semilla 42**, siguiendo el ejemplo del README de datos.

El tamaño y la antigüedad del dataset limitan las conclusiones: sirve para practicar MLOps, no para demostrar validez clínica.

## Requisitos mínimos

### 1. Proyecto y reproducibilidad

- Trabajar **por parejas**, con un único repositorio de GitHub.
- Separar código fuente, configuración, notebooks y resultados.
- Indicar versión de Python y dependencias con versiones fijadas.
- Proporcionar en el README los comandos para preparar el entorno, entrenar, realizar inferencia y generar el informe de drift.
- Identificar los datos utilizados mediante su fuente, nombre de archivo y hash verificable. No se requiere DVC.
- Realizar un EDA breve y comprobar columnas, tipos, valores ausentes y duplicados. Evitar que registros duplicados queden repartidos entre entrenamiento y test.

### 2. Artefacto y experimentación

- Entrenar una línea base sencilla, como `DummyClassifier`, y **al menos un modelo supervisado**, por ejemplo regresión logística o Random Forest.
- Guardar el preprocesamiento y el modelo en un único pipeline cargable, por ejemplo con Scikit-learn y Joblib.
- Ajustar las transformaciones solo con entrenamiento. Si se comparan configuraciones, utilizar validación interna o validación cruzada dentro de entrenamiento; reservar test para la evaluación final.
- Registrar al menos dos ejecuciones del modelo con alguna variación de configuración, conservando parámetros, métricas e identificador del artefacto. Se puede utilizar MLflow local o un registro estructurado en CSV/JSON.
- Evaluar con recall de la clase positiva, precision, F1-score y matriz de confusión. Justificar el equilibrio entre falsos negativos y falsos positivos, comparándolo con la línea base.

No se exige optimización exhaustiva, Deep Learning ni un umbral mínimo de rendimiento.

### 3. Inferencia y serving

- Implementar un comando de **inferencia batch** que lea un CSV y genere las predicciones.
- Crear una **API HTTP sencilla**, con `POST /predict` y `GET /health`. Se recomienda FastAPI.
- Reutilizar el mismo pipeline guardado en ambas vías, sin volver a entrenar al realizar predicciones.
- Devolver la clase, la probabilidad de la clase positiva y una versión identificable del modelo; en batch, conservar la correspondencia con las filas de entrada.
- Documentar un ejemplo válido y otro inválido. Validar las entradas y mostrar errores comprensibles ante columnas o campos incorrectos, valores no admitidos o un artefacto no disponible.

Es suficiente ejecutar la API en local. No se exige Docker, despliegue en la nube ni recuperación automatizada de versiones.

### 4. Monitorización y drift

- Preparar una referencia a partir de entrenamiento y un lote «actual» de simulación, generado sobre una copia de datos reservados.
- Introducir un cambio controlado en una o dos variables del lote actual, sin modificar el dataset original ni el test utilizado para la evaluación.
- Comparar referencia y actual mediante estadísticas y gráficos, usando Evidently o una implementación sencilla equivalente. Explicitar el método y el criterio utilizado para señalar cambios.
- Generar un informe que identifique el cambio, interprete su alcance y proponga una actuación.
- Conservar un registro básico de inferencia con fecha, versión del modelo, número de registros procesados y errores, sin almacenar los valores clínicos de las entradas.

El informe debe explicar que **drift no equivale necesariamente a pérdida de rendimiento**. Sin etiquetas no puede medirse esa pérdida, y una alerta no justifica reentrenar automáticamente. También se tendrán en cuenta las limitaciones del tamaño de muestra.

### 5. Calidad y trazabilidad

- Mantener un historial Git que permita entender la colaboración de ambos integrantes.
- Separar parámetros y rutas del código; evitar rutas absolutas del ordenador de los autores.
- Relacionar el artefacto con su configuración, versión de datos y commit de código mediante un pequeño archivo de metadatos o el registro de experimentos.
- Incluir comprobaciones automatizadas de carga del pipeline, predicción válida y rechazo de una entrada inválida.
- No guardar credenciales ni datos personales adicionales en el repositorio.

No se requiere configurar CI/CD, un registro de modelos ni una infraestructura de monitorización permanente.

### 6. Comunicación y reflexión

- Preparar una **demo de unos 5 minutos** que muestre el flujo principal: artefacto cargable, predicción batch, petición HTTP e informe de drift.
- Explicar las decisiones principales, una limitación relevante y el siguiente paso que se priorizaría.
- Documentar el reparto de responsabilidades. Ambos integrantes deberán comprender el conjunto de la solución.

## Criterios de evaluación

La práctica se realizará por parejas, que podrán formarse por elección propia. Si alguna persona no encuentra compañero, deberá comunicarlo al profesor.

La evaluación partirá de la entrega conjunta. Se podrá ajustar individualmente según las contribuciones y la comprensión demostrada, utilizando el historial del repositorio, el reparto de tareas y la explicación técnica. El número de commits no se considerará por sí solo una medida de participación.

| Criterio | Porcentaje | Evidencia esperada |
|---|---:|---|
| **Proyecto y reproducibilidad** | **20%** | Código organizado, dependencias y README ejecutable |
| **Artefacto y experimentación** | **20%** | Pipeline cargable; parámetros y métricas trazables |
| **Inferencia y serving** | **20%** | Batch/API, interfaz HTTP y errores comprensibles |
| **Monitorización y drift** | **15%** | Referencia/actual e interpretación con límites |
| **Calidad y trazabilidad** | **15%** | Git, configuración separada y versión identificable |
| **Comunicación y reflexión** | **10%** | Demo breve, decisiones, limitación y siguiente paso |

La rúbrica suma el **100% de la nota, sin bonus ni puntuación adicional**. Se valora la calidad de las evidencias, no la cantidad de tecnologías incorporadas.

## Entregables obligatorios

### Repositorio de GitHub

Un repositorio accesible para el profesor, con:

- Código de entrenamiento, inferencia batch, API y análisis de drift.
- Dependencias, configuración y comprobaciones automatizadas.
- Pipeline guardado o instrucciones para regenerarlo, metadatos y registro de experimentos.
- Ejemplos de entrada y resultados, incluido el informe de drift.
- README con autores, fuente de datos, preparación del entorno y comandos de ejecución.

El proyecto debe poder ejecutarse desde una copia limpia, sin depender de archivos inaccesibles o pasos manuales no documentados.

### Informe breve y demo

El informe recogerá roles, diseño del flujo, resultados, decisiones, interpretación del drift, una limitación y un siguiente paso. **Puede integrarse en el README** para evitar duplicar documentación.

La demo, en directo o grabada según indique el profesor, seguirá los comandos documentados. No es necesario preparar una presentación adicional.

Cada integrante subirá a PontIA **el mismo PDF**, con el informe o exportación del README, los nombres de ambos autores, la URL del repositorio y el commit de entrega. Si la demo es grabada, se incluirá también un enlace accesible.

## Conclusión

La práctica busca pasar de un modelo experimental a una solución pequeña pero reproducible: entrenar, guardar, cargar, servir y observar cambios. El resultado debe ser comprensible para otra persona y permitir explicar qué se ha construido, con qué versión y cuáles son sus límites.

## Anexo: criterios de evaluación

### 1. Proyecto y reproducibilidad — 20%

Se evalúa que una persona pueda preparar el entorno y ejecutar el proyecto siguiendo únicamente el README. Se comprobarán la organización, las dependencias, la identificación de los datos y la reproducibilidad del entrenamiento.

### 2. Artefacto y experimentación — 20%

Se evalúa que el pipeline guardado incluya el preprocesamiento, pueda cargarse desde otro proceso y disponga de parámetros y métricas trazables. Se revisarán la comparación con la línea base y la separación correcta entre entrenamiento y test.

### 3. Inferencia y serving — 20%

Se evalúan la inferencia batch y la interfaz HTTP, la reutilización del artefacto, el contrato de entrada/salida y los errores. La evidencia incluirá una predicción válida y el rechazo comprensible de una entrada inválida.

### 4. Monitorización y drift — 15%

Se evalúa la comparación entre referencia y datos actuales simulados, la explicación del método y la interpretación de los resultados. Se valorará distinguir cambios en las entradas de degradación predictiva y reconocer los límites del análisis.

### 5. Calidad y trazabilidad — 15%

Se evalúan el uso de Git, la configuración separada, las comprobaciones automatizadas y la relación identificable entre código, datos, ejecución y artefacto. Debe existir evidencia de participación de ambos integrantes.

### 6. Comunicación y reflexión — 10%

Se evalúa una demo breve y clara, la justificación de las decisiones y la capacidad de explicar una limitación y un siguiente paso concreto. No se requiere una memoria extensa.

## FAQs: preguntas y respuestas

### ¿Podemos elegir otro caso de uso o hacerlo individualmente?

No. El caso de uso y los datos son comunes para todos, y el trabajo se realiza por parejas. El CSV se incluye en `data/alt/heart_disease.csv`; la carpeta se llama `alt` por compatibilidad con el material, no porque el dataset sea opcional. Las prácticas guiadas mantienen Churn, pero el proyecto final utiliza Heart Disease.

### ¿Basta con un notebook?

No. Puede utilizarse para explorar, pero entrenamiento, inferencia y análisis de drift deben poder ejecutarse mediante comandos documentados.

### ¿Cuántos modelos y herramientas necesitamos?

Una línea base y un modelo supervisado, con al menos dos ejecuciones de configuración distinta del modelo. No se exige comparar muchas familias. MLflow y Evidently son opciones recomendadas, no obligatorias: un registro estructurado y un análisis sencillo bien justificado son suficientes.

### ¿Debemos implementar batch y API?

Sí, como dos interfaces pequeñas sobre el mismo pipeline: un comando que procese un CSV y una API local con `/predict` y `/health`. No se necesitan interfaz visual, autenticación ni servicios distribuidos.

### ¿Son obligatorios Docker, CI/CD o la nube?

No. Tampoco se exige rollback ni reentrenamiento automático. El alcance mínimo es local y debe poder reproducirse siguiendo el README.

### ¿Hay bonus por añadir más funcionalidades?

No. La nota se obtiene exclusivamente con los seis criterios de la rúbrica.

### ¿Qué ocurre si las métricas del modelo son modestas?

No existe un mínimo de rendimiento exigido. Se valoran el procedimiento correcto, la comparación con la línea base y la interpretación honesta de los resultados, sin presentar el sistema como una herramienta clínica.

### ¿Hay que redactar un informe aparte del README?

No. Si el README contiene toda la información solicitada, basta con exportarlo a PDF para la entrega en PontIA.
