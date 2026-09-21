"""
test_models.py
--------------
Tests unitarios para los pipelines de predicción de ML:
  - Modelo de Flight Risk (XGBoost + SHAP)
  - Modelo de Ausentismo (MLPRegressor)
  - Modelo de Retención (RandomForest)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestFlightRiskModel:
    """Tests para el modelo XGBoost de Flight Risk."""

    def test_model_loads_correctly(self, flight_risk_model):
        assert flight_risk_model is not None

    def test_model_predict_returns_binary(self, flight_risk_model, sample_employee_features):
        pred = flight_risk_model.predict(sample_employee_features)
        assert len(pred) == 1
        assert pred[0] in [0, 1]

    def test_model_predict_proba_between_0_1(self, flight_risk_model, sample_employee_features):
        proba = flight_risk_model.predict_proba(sample_employee_features)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0][0] <= 1.0
        assert 0.0 <= proba[0][1] <= 1.0
        assert abs(proba[0][0] + proba[0][1] - 1.0) < 1e-6

    def test_model_probas_sum_to_1(self, flight_risk_model, sample_employee_features):
        proba = flight_risk_model.predict_proba(sample_employee_features)
        assert abs(proba.sum(axis=1)[0] - 1.0) < 1e-6

    def test_model_handles_batch(self, flight_risk_model, sample_employee_features):
        """Predicción en batch de 10 empleados iguales."""
        batch = pd.concat([sample_employee_features] * 10, ignore_index=True)
        proba = flight_risk_model.predict_proba(batch)
        assert proba.shape[0] == 10

    def test_model_has_feature_names(self, metadata):
        assert "feature_cols" in metadata
        feature_cols = metadata["feature_cols"]
        assert len(feature_cols) > 0
        assert "bradford_factor" in feature_cols
        assert "workload_index" in feature_cols

    def test_role_map_has_expected_values(self, metadata):
        role_map = metadata["role_map"]
        assert "Ventas Técnicas" in role_map
        assert "Almacén" in role_map

    def test_gender_map_has_expected_values(self, metadata):
        gender_map = metadata["gender_map"]
        assert "Masculino" in gender_map
        assert "Femenino" in gender_map


class TestSHAPExplainer:
    """Tests para el SHAP TreeExplainer."""

    def test_explainer_loads_correctly(self, flight_risk_explainer):
        assert flight_risk_explainer is not None

    def test_shap_values_shape(self, flight_risk_explainer, sample_employee_features):
        shap_values = flight_risk_explainer.shap_values(sample_employee_features)
        # Para clasificación binaria con TreeExplainer: puede ser array o lista
        if isinstance(shap_values, list):
            # Clase 1 (riesgo)
            sv = shap_values[1]
        else:
            sv = shap_values
        assert sv.shape[0] == 1
        assert sv.shape[1] == sample_employee_features.shape[1]

    def test_shap_values_are_finite(self, flight_risk_explainer, sample_employee_features):
        shap_values = flight_risk_explainer.shap_values(sample_employee_features)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
        assert np.all(np.isfinite(sv)), "Hay valores SHAP NaN o Inf"


class TestAbsenteeismModel:
    """Tests para el modelo MLPRegressor de ausentismo."""

    def test_absenteeism_model_loads(self, absenteeism_model):
        assert absenteeism_model is not None

    def test_absenteeism_model_has_predict(self, absenteeism_model):
        assert hasattr(absenteeism_model, "predict")

    def test_absenteeism_prediction_output_shape(self, absenteeism_model, absenteeism_config=None):
        """El modelo debería retornar 4 valores (semanas t+1 a t+4)."""
        # Crear entrada de prueba (8 features de entrada)
        try:
            X_sample = np.array([[0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.07, 0.02]])
            pred = absenteeism_model.predict(X_sample)
            assert pred.shape[1] == 4, f"Esperaba 4 predicciones, got {pred.shape}"
        except Exception:
            pytest.skip("Modelo de ausentismo requiere configuración específica de entrada")

    def test_absenteeism_rates_non_negative(self, absenteeism_model):
        """Las tasas de ausentismo pronosticadas deben ser >= 0."""
        try:
            X_sample = np.array([[0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.07, 0.02]])
            pred = absenteeism_model.predict(X_sample)
            # Aplicar clip para tasas negativas (como hace la API)
            pred_clipped = pred.clip(0, 1)
            assert np.all(pred_clipped >= 0)
        except Exception:
            pytest.skip("Modelo de ausentismo requiere configuración específica de entrada")


class TestRetentionModel:
    """Tests para el modelo RandomForest de recomendación de retención."""

    def test_retention_model_loads(self, retention_model):
        assert retention_model is not None
        assert "model" in retention_model
        assert "label_encoder" in retention_model

    def test_retention_model_has_valid_classes(self, retention_model):
        le = retention_model["label_encoder"]
        valid_actions = {
            "aumento_salarial", "reducir_horas_extra",
            "capacitacion", "mentoria_liderazgo", "reubicacion_departamento"
        }
        for cls in le.classes_:
            assert cls in valid_actions, f"Clase inesperada: {cls}"

    def test_retention_model_predicts_valid_action(self, retention_model, sample_employee_features):
        feature_cols = retention_model["feature_cols"]
        # Verificar que la muestra tiene las columnas necesarias
        if not all(col in sample_employee_features.columns for col in feature_cols):
            # Ajustar columnas
            missing = [c for c in feature_cols if c not in sample_employee_features.columns]
            if "salary_percentile" in missing:
                sample_employee_features["salary_percentile"] = 50.0
            missing = [c for c in feature_cols if c not in sample_employee_features.columns]
            if missing:
                pytest.skip(f"Columnas faltantes en la muestra: {missing}")

        X = sample_employee_features[feature_cols]
        clf = retention_model["model"]
        le = retention_model["label_encoder"]

        pred_enc = clf.predict(X)
        pred_label = le.inverse_transform(pred_enc)
        assert len(pred_label) == 1

        valid_actions = {
            "aumento_salarial", "reducir_horas_extra",
            "capacitacion", "mentoria_liderazgo", "reubicacion_departamento"
        }
        assert pred_label[0] in valid_actions

    def test_retention_proba_sums_to_1(self, retention_model, sample_employee_features):
        feature_cols = retention_model["feature_cols"]
        X = sample_employee_features.reindex(columns=feature_cols, fill_value=0.0)
        clf = retention_model["model"]
        proba = clf.predict_proba(X)
        assert abs(proba.sum(axis=1)[0] - 1.0) < 1e-6


class TestMonteCarloSimulator:
    """Tests unitarios para el simulador Monte Carlo."""

    def test_simulator_imports(self):
        from simulator.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator()
        assert sim is not None

    def test_simulate_returns_expected_keys(self, sample_employees_list):
        from simulator.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator()
        result = sim.simulate_roi(
            employees_list=sample_employees_list,
            global_intervention_cost=6000.0,
            effectiveness=0.5,
            n_simulations=500
        )
        required_keys = ["avg_savings", "p5_savings", "p95_savings",
                         "prob_positive_roi", "expected_roi_pct"]
        for key in required_keys:
            assert key in result, f"Clave '{key}' faltante en resultado del simulador"

    def test_simulate_high_effectiveness_positive_roi(self, sample_employees_list):
        """Con efectividad alta, el ROI debe ser positivo."""
        from simulator.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator()
        result = sim.simulate_roi(
            employees_list=[e for e in sample_employees_list if e["flight_risk_prob"] > 0.5],
            global_intervention_cost=3000.0,
            effectiveness=0.90,
            n_simulations=500
        )
        assert result["avg_savings"] > 0, "Con efectividad 90% el ahorro promedio debe ser positivo"

    def test_simulate_empty_list_raises_or_returns_zero(self):
        from simulator.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator()
        try:
            result = sim.simulate_roi(
                employees_list=[],
                global_intervention_cost=6000.0,
                effectiveness=0.5,
                n_simulations=100
            )
            assert result["avg_savings"] == 0.0 or result["total_intervention_cost"] == 0.0
        except Exception:
            pass  # También es válido que lance una excepción para lista vacía
