"""Interfaz Streamlit: formulario que consume la API por HTTP.

La UI NO carga el modelo ni calcula predicciones: envía la fila a POST /predict
y muestra lo que devuelve la API (probabilidad, clase, umbral y versión).

Arranque (con la API ya levantada, en otra terminal):

    uv run streamlit run src/mlops_course/ui.py --server.address 127.0.0.1 \
        --server.port 8501

`API_URL` permite apuntar a otra dirección (por defecto http://127.0.0.1:8000).
"""

import os

import requests
import streamlit as st
from pydantic import ValidationError

from mlops_course.schemas import Prediccion

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = (3, 15)  # (conexión, lectura) en segundos

st.set_page_config(page_title="Predicción de abandono", page_icon="📉")
st.title("Predicción de abandono")
st.caption(f"API: {API_URL}")

# st.form evita llamar a la API con cada cambio de widget: solo al pulsar el botón.
with st.form("cliente"):
    antiguedad = st.number_input(
        "Antigüedad (meses)", min_value=0, max_value=120, value=12
    )
    cuota = st.number_input(
        "Cuota mensual (€)", min_value=0.01, max_value=500.0, value=70.35
    )
    contrato = st.selectbox("Tipo de contrato", ["mensual", "anual", "bianual"])
    internet = st.selectbox("Tipo de internet", ["dsl", "fibra", "sin_internet"])
    pago = st.selectbox(
        "Método de pago",
        ["cheque_electronico", "cheque_postal", "transferencia", "tarjeta"],
    )
    enviado = st.form_submit_button("Predecir")

if enviado:
    payload = {
        "antiguedad_meses": int(antiguedad),
        "cuota_mensual": float(cuota),
        "tipo_contrato": contrato,
        "tipo_internet": internet,
        "metodo_pago": pago,
    }
    # TODO S3: llamar a la API y mostrar el resultado.
    #   1. requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
    #      y response.raise_for_status().
    #   2. Capturar por separado requests.Timeout, requests.ConnectionError,
    #      requests.HTTPError (distinguir 422 del resto) y RequestException,
    #      mostrando st.error(...) SIN inventar ninguna predicción.
    #   3. Validar la respuesta con Prediccion.model_validate(response.json());
    #      si falla (JSONDecodeError o ValidationError), mostrar un error.
    #   4. Si todo va bien: st.metric con la probabilidad en %, st.progress,
    #      la clase predicha, y st.caption con umbral y model_version.
    st.warning("TODO S3: completar la llamada a la API en ui.py")
