"""
cohort_analysis.py
------------------
Análisis de cohortes de empleados segmentados por antigüedad y departamento.
Calcula tasa de rotación, flight risk promedio, Bradford Factor y NOM-035 por cohorte.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Any


# Definición de cohortes de antigüedad (en meses)
TENURE_COHORTS = [
    (0, 6,    "0-6 meses"),
    (6, 12,   "6-12 meses"),
    (12, 24,  "1-2 años"),
    (24, 60,  "2-5 años"),
    (60, 999, "5+ años"),
]


def _assign_cohort(tenure_months: float) -> str:
    """Asigna etiqueta de cohorte según antigüedad en meses."""
    for lo, hi, label in TENURE_COHORTS:
        if lo <= tenure_months < hi:
            return label
    return "5+ años"


def get_cohort_analysis(db_path: str, flight_risk_probs: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Genera el análisis de cohortes de empleados.

    Parámetros
    ----------
    db_path : str
        Ruta a la base de datos SQLite.
    flight_risk_probs : dict, opcional
        Diccionario {employee_id: prob} con probabilidades de riesgo precalculadas.
        Si no se proporciona, la columna flight_risk_prob será None.

    Retorna
    -------
    List[Dict] con una entrada por cada (cohorte, departamento) con métricas agregadas.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Cargar empleados con NOM-035
        query = """
        SELECT
            e.employee_id, e.role, e.tenure_months, e.has_quit,
            n.workload_index, n.leadership_index, n.organizational_environment_index,
            e.monthly_salary, e.overtime_hours_last_month
        FROM employees e
        LEFT JOIN nom035_survey n ON e.employee_id = n.employee_id
        """
        df = pd.read_sql_query(query, conn)

        # Calcular Bradford Factor por empleado
        events_df = pd.read_sql_query(
            "SELECT employee_id, duration_days FROM absenteeism_events", conn
        )
        bf = (
            events_df.groupby("employee_id")
            .agg(S=("duration_days", "count"), D=("duration_days", "sum"))
            .assign(bf=lambda x: x["S"] ** 2 * x["D"])["bf"]
        )
        df = df.join(bf.rename("bradford_factor"), on="employee_id")
        df["bradford_factor"] = df["bradford_factor"].fillna(0.0)

        # Asignar flight risk si se proporcionó
        if flight_risk_probs:
            df["flight_risk_prob"] = df["employee_id"].map(flight_risk_probs)
        else:
            df["flight_risk_prob"] = np.nan

        # Asignar cohorte de antigüedad
        df["cohort"] = df["tenure_months"].apply(_assign_cohort)

        # Definir orden categórico para cohortes
        cohort_order = [label for _, _, label in TENURE_COHORTS]
        df["cohort"] = pd.Categorical(df["cohort"], categories=cohort_order, ordered=True)

        # Agregar por (cohorte, rol/departamento)
        results = []
        for (cohort, role), grp in df.groupby(["cohort", "role"], observed=True):
            turnover_rate = grp["has_quit"].mean() * 100
            avg_risk = grp["flight_risk_prob"].mean() if not grp["flight_risk_prob"].isna().all() else None
            results.append({
                "cohort": str(cohort),
                "role": str(role),
                "headcount": int(len(grp)),
                "active": int((grp["has_quit"] == 0).sum()),
                "quit": int(grp["has_quit"].sum()),
                "turnover_rate_pct": round(float(turnover_rate), 1),
                "avg_flight_risk_pct": round(float(avg_risk) * 100, 1) if avg_risk is not None else None,
                "avg_bradford_factor": round(float(grp["bradford_factor"].mean()), 1),
                "avg_workload_index": round(float(grp["workload_index"].mean()), 1),
                "avg_leadership_index": round(float(grp["leadership_index"].mean()), 1),
                "avg_salary": round(float(grp["monthly_salary"].mean()), 2),
                "avg_overtime_hours": round(float(grp["overtime_hours_last_month"].mean()), 1),
            })

        # Ordenar por cohorte (orden natural) y rol
        results.sort(key=lambda x: (cohort_order.index(x["cohort"]), x["role"]))
        return results

    finally:
        conn.close()


def get_department_risk_heatmap(db_path: str, flight_risk_probs: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Genera datos para un heatmap de riesgo por departamento vs cohorte de antigüedad.
    Útil para identificar el segmento más vulnerable de la organización.
    """
    rows = get_cohort_analysis(db_path, flight_risk_probs)
    return [
        {
            "cohort": r["cohort"],
            "role": r["role"],
            "turnover_rate_pct": r["turnover_rate_pct"],
            "avg_flight_risk_pct": r["avg_flight_risk_pct"],
            "headcount": r["headcount"],
        }
        for r in rows
    ]
