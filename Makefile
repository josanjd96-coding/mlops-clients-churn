# Atajos OPCIONALES. Cada objetivo muestra el comando real que ejecuta.
# Sin Make (p. ej. Windows), copia el comando de la derecha en la terminal.
# Equivalencias completas en README.md.

ESCENARIO ?= covariables

.PHONY: help install lint format train search mlflow batch api ui \
	monitor test smoke-test clean

help:                       ## Lista de objetivos
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ ->/'

install:                    ## uv sync
	uv sync

lint:                       ## uv run ruff check . && uv run ruff format --check .
	uv run ruff check .
	uv run ruff format --check .

format:                     ## uv run ruff format . && uv run ruff check --fix .
	uv run ruff format .
	uv run ruff check --fix .

train:                      ## uv run python -m mlops_course.train
	uv run python -m mlops_course.train

search:                     ## uv run python -m mlops_course.train --busqueda
	uv run python -m mlops_course.train --busqueda

mlflow:                     ## uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

batch:                      ## uv run python -m mlops_course.predict
	uv run python -m mlops_course.predict

api:                        ## uv run uvicorn mlops_course.api:app --host 127.0.0.1 --port 8000
	uv run uvicorn mlops_course.api:app --host 127.0.0.1 --port 8000

ui:                         ## uv run streamlit run src/mlops_course/ui.py ...
	uv run streamlit run src/mlops_course/ui.py --server.address 127.0.0.1 --server.port 8501

monitor:                    ## simular + informe + comparar (ESCENARIO=estable|covariables|concepto)
	uv run python -m mlops_course.monitoring simular --escenario $(ESCENARIO)
	uv run python -m mlops_course.monitoring informe
	uv run python -m mlops_course.monitoring comparar

test:                       ## uv run pytest
	uv run pytest

smoke-test:                 ## uv run python scripts/smoke_test.py
	uv run python scripts/smoke_test.py

clean:                      ## borra artefactos, informes y la base de MLflow locales
	rm -rf artifacts reports mlruns mlartifacts mlflow.db
