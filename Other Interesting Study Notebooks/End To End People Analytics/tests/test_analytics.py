"""
test_analytics.py
-----------------
Tests unitarios para los módulos del paquete analytics/:
  - Análisis de cohortes
  - Análisis de supervivencia Kaplan-Meier
  - Detección de anomalías (IsolationForest)
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestCohortAnalysis:
    """Tests para analytics/cohort_analysis.py"""

    def test_get_cohort_analysis_returns_list(self, db_path):
        from analytics.cohort_analysis import get_cohort_analysis
        result = get_cohort_analysis(db_path)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_cohort_has_required_keys(self, db_path):
        from analytics.cohort_analysis import get_cohort_analysis
        result = get_cohort_analysis(db_path)
        required_keys = [
            "cohort", "role", "headcount", "active", "quit",
            "turnover_rate_pct", "avg_bradford_factor",
            "avg_workload_index", "avg_leadership_index"
        ]
        for cohort in result:
            for key in required_keys:
                assert key in cohort, f"Clave '{key}' faltante en cohort: {cohort}"

    def test_cohort_turnover_rate_valid_range(self, db_path):
        from analytics.cohort_analysis import get_cohort_analysis
        result = get_cohort_analysis(db_path)
        for cohort in result:
            rate = cohort["turnover_rate_pct"]
            assert 0.0 <= rate <= 100.0, f"Tasa de rotación fuera de rango: {rate}"

    def test_cohort_headcount_equals_active_plus_quit(self, db_path):
        from analytics.cohort_analysis import get_cohort_analysis
        result = get_cohort_analysis(db_path)
        for cohort in result:
            assert cohort["headcount"] == cohort["active"] + cohort["quit"], \
                f"Headcount inconsistente en cohorte: {cohort['cohort']}"

    def test_all_cohorts_represented(self, db_path):
        from analytics.cohort_analysis import get_cohort_analysis, TENURE_COHORTS
        result = get_cohort_analysis(db_path)
        cohort_labels = {r["cohort"] for r in result}
        expected_labels = {label for _, _, label in TENURE_COHORTS}
        # Al menos la mayoría de las cohortes deben estar presentes
        assert len(cohort_labels.intersection(expected_labels)) >= 3

    def test_roles_are_valid(self, db_path):
        from analytics.cohort_analysis import get_cohort_analysis
        result = get_cohort_analysis(db_path)
        valid_roles = {"Almacén", "Ventas Técnicas"}
        for cohort in result:
            assert cohort["role"] in valid_roles, f"Rol inválido: {cohort['role']}"

    def test_with_flight_risk_probs(self, db_path):
        from analytics.cohort_analysis import get_cohort_analysis
        # Simular probabilidades de riesgo para todos los empleados
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT employee_id FROM employees LIMIT 50")
        ids = [r[0] for r in cur.fetchall()]
        conn.close()

        fake_probs = {emp_id: 0.5 for emp_id in ids}
        result = get_cohort_analysis(db_path, flight_risk_probs=fake_probs)
        assert isinstance(result, list)

    def test_heatmap_function_returns_data(self, db_path):
        from analytics.cohort_analysis import get_department_risk_heatmap
        result = get_department_risk_heatmap(db_path)
        assert isinstance(result, list)
        for entry in result:
            assert "cohort" in entry
            assert "role" in entry
            assert "turnover_rate_pct" in entry


class TestSurvivalAnalysis:
    """Tests para analytics/survival_analysis.py"""

    def test_survival_returns_dict(self, db_path):
        from analytics.survival_analysis import get_survival_curves
        result = get_survival_curves(db_path)
        assert isinstance(result, dict)

    def test_survival_has_curves_key(self, db_path):
        from analytics.survival_analysis import get_survival_curves
        result = get_survival_curves(db_path)
        assert "curves" in result
        assert "median_survival" in result
        assert "summary" in result

    def test_survival_has_global_curve(self, db_path):
        from analytics.survival_analysis import get_survival_curves
        result = get_survival_curves(db_path)
        group_names = [c["group"] for c in result["curves"]]
        assert "Global" in group_names

    def test_survival_probabilities_decreasing(self, db_path):
        """La probabilidad de supervivencia debe ser monotónicamente no-creciente."""
        from analytics.survival_analysis import get_survival_curves
        result = get_survival_curves(db_path)
        global_curve = next(c for c in result["curves"] if c["group"] == "Global")
        probs = [p["survival_prob"] for p in global_curve["points"]]
        for i in range(1, len(probs)):
            assert probs[i] <= probs[i-1] + 1e-6, \
                f"Probabilidad de supervivencia no es monotónica en índice {i}: {probs[i-1]} -> {probs[i]}"

    def test_survival_probs_between_0_and_1(self, db_path):
        from analytics.survival_analysis import get_survival_curves
        result = get_survival_curves(db_path)
        for curve in result["curves"]:
            for point in curve["points"]:
                assert 0.0 <= point["survival_prob"] <= 1.0 + 1e-6

    def test_survival_at_months(self, db_path):
        from analytics.survival_analysis import get_survival_at_months
        result = get_survival_at_months(db_path, months=[12, 24])
        assert isinstance(result, list)
        assert len(result) > 0
        for row in result:
            assert "group" in row
            assert "month" in row
            assert "survival_prob" in row

    def test_median_survival_is_positive_or_none(self, db_path):
        from analytics.survival_analysis import get_survival_curves
        result = get_survival_curves(db_path)
        for group, median in result["median_survival"].items():
            if median is not None:
                assert median > 0, f"Mediana de supervivencia negativa para {group}: {median}"

    def test_summary_has_total_employees(self, db_path):
        from analytics.survival_analysis import get_survival_curves
        result = get_survival_curves(db_path)
        summary = result["summary"]
        assert "total_employees" in summary
        assert summary["total_employees"] > 0


class TestAnomalyDetection:
    """Tests para analytics/anomaly_detection.py"""

    def test_detect_anomalies_returns_dict(self, db_path):
        from analytics.anomaly_detection import detect_anomalies
        result = detect_anomalies(db_path, contamination=0.1)
        assert isinstance(result, dict)

    def test_detect_anomalies_has_required_keys(self, db_path):
        from analytics.anomaly_detection import detect_anomalies
        result = detect_anomalies(db_path)
        assert "anomalies" in result
        assert "summary" in result
        assert "feature_importance" in result

    def test_anomalies_list_is_not_empty(self, db_path):
        from analytics.anomaly_detection import detect_anomalies
        result = detect_anomalies(db_path)
        assert len(result["anomalies"]) > 0

    def test_anomaly_labels_are_valid(self, db_path):
        from analytics.anomaly_detection import detect_anomalies
        result = detect_anomalies(db_path)
        valid_labels = {"NORMAL", "SOSPECHOSO", "ANOMALÍA CRÍTICA"}
        for emp in result["anomalies"]:
            assert emp["anomaly_label"] in valid_labels

    def test_anomaly_percentile_between_0_and_100(self, db_path):
        from analytics.anomaly_detection import detect_anomalies
        result = detect_anomalies(db_path)
        for emp in result["anomalies"]:
            assert 0.0 <= emp["anomaly_percentile"] <= 100.0

    def test_summary_counts_add_up(self, db_path):
        from analytics.anomaly_detection import detect_anomalies
        result = detect_anomalies(db_path)
        summary = result["summary"]
        total = summary["normal"] + summary["suspicious"] + summary["critical"]
        assert total == summary["total_analyzed"]

    def test_feature_importance_sums_to_1(self, db_path):
        from analytics.anomaly_detection import detect_anomalies
        result = detect_anomalies(db_path)
        fi = result["feature_importance"]
        total = sum(fi.values())
        assert abs(total - 1.0) < 0.01, f"Importancia de features no suma 1.0: {total}"

    def test_contamination_affects_critical_count(self, db_path):
        from analytics.anomaly_detection import detect_anomalies
        result_low = detect_anomalies(db_path, contamination=0.05)
        result_high = detect_anomalies(db_path, contamination=0.20)
        # Con mayor contaminación, debe haber más anomalías críticas o sospechosas
        non_normal_low = result_low["summary"]["critical"] + result_low["summary"]["suspicious"]
        non_normal_high = result_high["summary"]["critical"] + result_high["summary"]["suspicious"]
        assert non_normal_high >= non_normal_low


class TestAlertEngine:
    """Tests para alerts/alert_engine.py"""

    def test_fire_alerts_returns_dict(self, sample_employees_list):
        from alerts.alert_engine import fire_alerts
        result = fire_alerts(sample_employees_list, threshold=0.65)
        assert isinstance(result, dict)

    def test_fire_alerts_has_required_keys(self, sample_employees_list):
        from alerts.alert_engine import fire_alerts
        result = fire_alerts(sample_employees_list, threshold=0.65)
        assert "high_risk_count" in result
        assert "email_sent" in result
        assert "slack_sent" in result
        assert "employees_alerted" in result

    def test_fire_alerts_counts_correctly(self, sample_employees_list):
        from alerts.alert_engine import fire_alerts
        threshold = 0.70
        result = fire_alerts(sample_employees_list, threshold=threshold)
        expected_count = sum(
            1 for e in sample_employees_list if e["flight_risk_prob"] >= threshold
        )
        assert result["high_risk_count"] == expected_count

    def test_no_alerts_below_threshold(self, sample_employees_list):
        from alerts.alert_engine import fire_alerts
        result = fire_alerts(sample_employees_list, threshold=0.99)
        assert result["high_risk_count"] == 0

    def test_all_above_threshold_0(self, sample_employees_list):
        from alerts.alert_engine import fire_alerts
        result = fire_alerts(sample_employees_list, threshold=0.0)
        assert result["high_risk_count"] == len(sample_employees_list)

    def test_email_not_sent_without_config(self, sample_employees_list):
        """Sin variables de entorno configuradas, el email no debe enviarse."""
        from alerts.alert_engine import fire_alerts
        result = fire_alerts(sample_employees_list, threshold=0.0)
        # Sin ALERT_EMAIL_FROM etc., debe ser False
        assert result["email_sent"] == False

    def test_slack_not_sent_without_webhook(self, sample_employees_list):
        """Sin Webhook de Slack configurado, la alerta no debe enviarse."""
        from alerts.alert_engine import fire_alerts
        result = fire_alerts(sample_employees_list, threshold=0.0)
        assert result["slack_sent"] == False
