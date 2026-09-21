"""
conftest.py
------------
Fixtures compartidas para toda la suite de tests de People Analytics.
"""

import os
import sqlite3
import pickle
import tempfile
import pytest
import pandas as pd
import numpy as np


# Ruta base del proyecto (2 niveles arriba de /tests/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "people_analytics.db")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def db_path():
    """Verifica que la base de datos exista y la retorna."""
    if not os.path.exists(DB_PATH):
        pytest.skip(f"Base de datos no encontrada en {DB_PATH}. Ejecuta data/load_db.py primero.")
    return DB_PATH


@pytest.fixture(scope="session")
def models_dir():
    return MODELS_DIR


@pytest.fixture(scope="session")
def flight_risk_model(models_dir):
    """Carga el modelo de Flight Risk desde disco."""
    path = os.path.join(models_dir, "flight_risk_model.pkl")
    if not os.path.exists(path):
        pytest.skip("Modelo flight_risk_model.pkl no encontrado. Ejecuta models/train_flight_risk.py.")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def flight_risk_explainer(models_dir):
    """Carga el SHAP explainer desde disco."""
    path = os.path.join(models_dir, "flight_risk_explainer.pkl")
    if not os.path.exists(path):
        pytest.skip("SHAP explainer no encontrado.")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def metadata(models_dir):
    """Carga los metadatos del modelo (feature_cols, mappings)."""
    path = os.path.join(models_dir, "metadata.pkl")
    if not os.path.exists(path):
        pytest.skip("Metadata.pkl no encontrado.")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def retention_model(models_dir):
    """Carga el modelo de retención desde disco."""
    path = os.path.join(models_dir, "retention_model.pkl")
    if not os.path.exists(path):
        pytest.skip("retention_model.pkl no encontrado. Ejecuta models/retention_model.py.")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def absenteeism_model(models_dir):
    """Carga el modelo de ausentismo desde disco."""
    path = os.path.join(models_dir, "absenteeism_model.pkl")
    if not os.path.exists(path):
        pytest.skip("absenteeism_model.pkl no encontrado.")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def sample_employee_features(metadata):
    """Genera un vector de features de muestra para un empleado de Ventas."""
    feature_cols = metadata["feature_cols"]
    # Valores típicos para un empleado de ventas de alto riesgo
    values = {
        "role_encoded": 1,          # Ventas Técnicas
        "age": 30,
        "gender_encoded": 0,        # Masculino
        "tenure_months": 8,
        "monthly_salary": 15000.0,
        "distance_km": 35.0,
        "overtime_hours_last_month": 25.0,
        "training_hours": 4.0,
        "last_performance_rating": 2,
        "leadership_index": 38.0,
        "workload_index": 78.0,
        "organizational_environment_index": 45.0,
        "quota_attainment_pct": 55.0,
        "sales_closed": 3,
        "commissions_earned": 2000.0,
        "on_time_delivery_pct": -1.0,
        "inventory_errors": -1,
        "machinery_incidents": -1,
        "bradford_factor": 75.0,
    }
    return pd.DataFrame([[values[col] for col in feature_cols]], columns=feature_cols)


@pytest.fixture(scope="session")
def sample_employees_list():
    """Lista de empleados de muestra para tests de Monte Carlo y alertas."""
    return [
        {"role": "Ventas Técnicas", "monthly_salary": 20000.0, "flight_risk_prob": 0.82},
        {"role": "Almacén", "monthly_salary": 12000.0, "flight_risk_prob": 0.71},
        {"role": "Ventas Técnicas", "monthly_salary": 18000.0, "flight_risk_prob": 0.45},
        {"role": "Almacén", "monthly_salary": 10000.0, "flight_risk_prob": 0.22},
    ]
