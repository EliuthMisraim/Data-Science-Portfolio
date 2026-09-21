"""
test_api.py
-----------
Tests de integración para los endpoints FastAPI de People Analytics.
Usa el TestClient de FastAPI/Starlette para simular peticiones HTTP.
"""

import os
import sys
import pytest

# Asegurar que el project root está en sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="module")
def client(db_path, models_dir):
    """Crea el TestClient de FastAPI con la app configurada."""
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests del endpoint raíz /"""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_api_status(self, client):
        data = client.get("/").json()
        assert "api_status" in data
        assert data["api_status"] == "online"

    def test_root_has_model_flags(self, client):
        data = client.get("/").json()
        assert "flight_risk_model_loaded" in data
        assert "absenteeism_model_loaded" in data
        assert "database_exists" in data


class TestDescriptiveStats:
    """Tests del endpoint /db/descriptive-stats"""

    def test_descriptive_stats_returns_200(self, client):
        response = client.get("/db/descriptive-stats")
        assert response.status_code == 200

    def test_descriptive_stats_has_expected_keys(self, client):
        data = client.get("/db/descriptive-stats").json()
        required_keys = ["total_employees", "total_quit", "general_turnover_rate",
                         "dept_turnover", "nom035_averages", "performance_averages"]
        for key in required_keys:
            assert key in data, f"Falta la clave '{key}' en /db/descriptive-stats"

    def test_total_employees_positive(self, client):
        data = client.get("/db/descriptive-stats").json()
        assert data["total_employees"] > 0

    def test_turnover_rate_between_0_and_100(self, client):
        data = client.get("/db/descriptive-stats").json()
        rate = data["general_turnover_rate"]
        assert 0.0 <= rate <= 100.0


class TestEmployeesRisk:
    """Tests del endpoint /db/employees-risk"""

    def test_employees_risk_returns_list(self, client):
        response = client.get("/db/employees-risk")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_employees_risk_not_empty(self, client):
        data = client.get("/db/employees-risk").json()
        assert len(data) > 0

    def test_each_employee_has_required_fields(self, client):
        data = client.get("/db/employees-risk").json()
        required_fields = ["employee_id", "name", "role", "monthly_salary",
                           "flight_risk_prob", "high_risk", "bradford_factor"]
        for emp in data[:5]:  # Verificar los primeros 5
            for field in required_fields:
                assert field in emp, f"Campo '{field}' faltante en empleado"

    def test_flight_risk_prob_between_0_and_1(self, client):
        data = client.get("/db/employees-risk").json()
        for emp in data:
            prob = emp["flight_risk_prob"]
            assert 0.0 <= prob <= 1.0, f"Probabilidad fuera de rango: {prob}"

    def test_sorted_by_risk_descending(self, client):
        data = client.get("/db/employees-risk").json()
        probs = [emp["flight_risk_prob"] for emp in data]
        assert probs == sorted(probs, reverse=True), "Empleados no están ordenados por riesgo descendente"


class TestFlightRiskByID:
    """Tests del endpoint /predict/flight-risk/{employee_id}"""

    def test_valid_employee_returns_prediction(self, client, db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT employee_id FROM employees WHERE has_quit=0 LIMIT 1")
        emp_id = cur.fetchone()[0]
        conn.close()

        response = client.get(f"/predict/flight-risk/{emp_id}")
        assert response.status_code == 200

    def test_invalid_employee_returns_404_or_500(self, client):
        response = client.get("/predict/flight-risk/NONEXISTENT_ID")
        assert response.status_code in [404, 500]

    def test_prediction_has_shap_values(self, client, db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT employee_id FROM employees WHERE has_quit=0 LIMIT 1")
        emp_id = cur.fetchone()[0]
        conn.close()

        data = client.get(f"/predict/flight-risk/{emp_id}").json()
        assert "shap_values" in data
        assert isinstance(data["shap_values"], dict)
        assert len(data["shap_values"]) > 0


class TestAbsenteeismHistory:
    """Tests del endpoint /db/absenteeism-history/{dept}"""

    def test_almacen_history_returns_data(self, client):
        response = client.get("/db/absenteeism-history/Almacén")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_ventas_history_returns_data(self, client):
        response = client.get("/db/absenteeism-history/Ventas Técnicas")
        assert response.status_code == 200

    def test_invalid_dept_returns_400(self, client):
        response = client.get("/db/absenteeism-history/DepartamentoInvalido")
        assert response.status_code == 400

    def test_history_has_absenteeism_rate(self, client):
        data = client.get("/db/absenteeism-history/Almacén").json()
        for record in data[:3]:
            assert "absenteeism_rate" in record
            assert 0.0 <= record["absenteeism_rate"] <= 1.0


class TestMonteCarloROI:
    """Tests del endpoint /simulate/roi"""

    def test_monte_carlo_returns_valid_response(self, client, sample_employees_list):
        payload = {
            "employees": sample_employees_list,
            "global_intervention_cost": 6000.0,
            "effectiveness": 0.5,
            "n_simulations": 500  # Reducido para velocidad en tests
        }
        response = client.post("/simulate/roi", json=payload)
        assert response.status_code == 200

    def test_monte_carlo_has_required_keys(self, client, sample_employees_list):
        payload = {
            "employees": sample_employees_list,
            "global_intervention_cost": 6000.0,
            "effectiveness": 0.5,
            "n_simulations": 500
        }
        data = client.post("/simulate/roi", json=payload).json()
        required_keys = ["avg_savings", "p5_savings", "p95_savings",
                         "prob_positive_roi", "expected_roi_pct"]
        for key in required_keys:
            assert key in data, f"Clave '{key}' faltante en respuesta de simulación"

    def test_prob_positive_roi_is_percentage(self, client, sample_employees_list):
        payload = {
            "employees": sample_employees_list,
            "global_intervention_cost": 6000.0,
            "effectiveness": 0.9,  # Alta efectividad → debe ser rentable
            "n_simulations": 500
        }
        data = client.post("/simulate/roi", json=payload).json()
        assert 0.0 <= data["prob_positive_roi"] <= 100.0


class TestAnalyticsEndpoints:
    """Tests de los nuevos endpoints analíticos."""

    def test_cohorts_endpoint_returns_200(self, client):
        response = client.get("/analytics/cohorts")
        assert response.status_code == 200

    def test_cohorts_has_expected_structure(self, client):
        data = client.get("/analytics/cohorts").json()
        assert "cohorts" in data
        assert isinstance(data["cohorts"], list)
        if len(data["cohorts"]) > 0:
            cohort = data["cohorts"][0]
            assert "cohort" in cohort
            assert "role" in cohort
            assert "turnover_rate_pct" in cohort

    def test_survival_curve_returns_200(self, client):
        response = client.get("/analytics/survival-curve")
        assert response.status_code == 200

    def test_survival_curve_has_groups(self, client):
        data = client.get("/analytics/survival-curve").json()
        assert "curves" in data
        assert len(data["curves"]) > 0
        # Verificar que hay curva global
        group_names = [c["group"] for c in data["curves"]]
        assert "Global" in group_names

    def test_anomalies_endpoint_returns_200(self, client):
        response = client.get("/analytics/anomalies")
        assert response.status_code == 200

    def test_anomalies_has_summary(self, client):
        data = client.get("/analytics/anomalies").json()
        assert "summary" in data
        assert "anomalies" in data
        summary = data["summary"]
        assert "total_analyzed" in summary
        assert "critical" in summary
        assert "suspicious" in summary

    def test_anomaly_labels_are_valid(self, client):
        data = client.get("/analytics/anomalies").json()
        valid_labels = {"NORMAL", "SOSPECHOSO", "ANOMALÍA CRÍTICA"}
        for emp in data["anomalies"][:10]:
            assert emp["anomaly_label"] in valid_labels


class TestRetentionEndpoint:
    """Tests del endpoint /predict/retention/{employee_id}"""

    def test_retention_prediction_returns_200(self, client, db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT employee_id FROM employees WHERE has_quit=0 LIMIT 1")
        emp_id = cur.fetchone()[0]
        conn.close()

        response = client.get(f"/predict/retention/{emp_id}")
        assert response.status_code == 200

    def test_retention_has_recommended_action(self, client, db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT employee_id FROM employees WHERE has_quit=0 LIMIT 1")
        emp_id = cur.fetchone()[0]
        conn.close()

        data = client.get(f"/predict/retention/{emp_id}").json()
        assert "recommended_action" in data
        valid_actions = {
            "aumento_salarial", "reducir_horas_extra",
            "capacitacion", "mentoria_liderazgo", "reubicacion_departamento"
        }
        assert data["recommended_action"] in valid_actions
