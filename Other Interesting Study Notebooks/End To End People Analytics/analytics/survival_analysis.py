"""
survival_analysis.py
---------------------
Análisis de supervivencia Kaplan-Meier para modelar tiempo hasta la renuncia.
Usa la librería `lifelines` para calcular curvas de supervivencia por rol/departamento.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def get_survival_curves(db_path: str) -> Dict[str, Any]:
    """
    Calcula curvas de supervivencia Kaplan-Meier para cada rol (Almacén / Ventas Técnicas)
    y para la organización en conjunto.

    Parámetros
    ----------
    db_path : str
        Ruta a la base de datos SQLite.

    Retorna
    -------
    Dict con:
        - curves: List[Dict] — puntos de la curva KM por grupo
        - median_survival: Dict {role: mediana_en_meses}
        - summary: métricas descriptivas de supervivencia
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        raise ImportError("La librería 'lifelines' no está instalada. Corre: pip install lifelines")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT employee_id, role, tenure_months, has_quit FROM employees",
            conn
        )
    finally:
        conn.close()

    # lifelines requiere: durations = tiempo observado, event_observed = 1 si ocurrió el evento
    curves = []
    median_survival = {}
    groups = [("Global", df)] + [(role, df[df["role"] == role]) for role in df["role"].unique()]

    for group_name, group_df in groups:
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group_df["tenure_months"],
            event_observed=group_df["has_quit"],
            label=group_name
        )

        # Extraer la curva como lista de puntos (timeline, survival_prob)
        km_df = kmf.survival_function_.reset_index()
        km_df.columns = ["timeline", "survival_prob"]

        # Intervalo de confianza 95%
        ci_df = kmf.confidence_interval_survival_function_.reset_index()
        ci_df.columns = ["timeline", "ci_lower", "ci_upper"]
        km_df = km_df.merge(ci_df, on="timeline")

        curves.append({
            "group": group_name,
            "points": [
                {
                    "month": round(float(row["timeline"]), 1),
                    "survival_prob": round(float(row["survival_prob"]), 4),
                    "ci_lower": round(float(row["ci_lower"]), 4),
                    "ci_upper": round(float(row["ci_upper"]), 4),
                }
                for _, row in km_df.iterrows()
            ]
        })

        # Mediana de supervivencia (mes en el que el 50% ya ha renunciado)
        median = kmf.median_survival_time_
        median_survival[group_name] = float(median) if not np.isnan(median) and not np.isinf(median) else None

    # Estadísticas descriptivas adicionales
    summary = {
        "total_employees": int(len(df)),
        "total_events": int(df["has_quit"].sum()),
        "overall_turnover_rate_pct": round(df["has_quit"].mean() * 100, 1),
        "median_tenure_active_months": round(
            float(df[df["has_quit"] == 0]["tenure_months"].median()), 1
        ),
        "roles_analyzed": [g for g, _ in groups if g != "Global"],
    }

    return {
        "curves": curves,
        "median_survival": median_survival,
        "summary": summary,
    }


def get_survival_at_months(db_path: str, months: List[int] = None) -> List[Dict[str, Any]]:
    """
    Calcula la probabilidad de supervivencia en puntos de tiempo específicos
    para cada grupo (útil para tablas de retención ejecutivas).

    Retorna lista de puntos {group, month, survival_prob}.
    """
    if months is None:
        months = [6, 12, 18, 24, 36, 48, 60]

    result = get_survival_curves(db_path)
    rows = []
    for curve in result["curves"]:
        group = curve["group"]
        # Interpolar la curva para los meses solicitados
        points_df = pd.DataFrame(curve["points"]).set_index("month")
        for m in months:
            # Buscar el punto más cercano anterior o igual
            valid = points_df[points_df.index <= m]
            if len(valid) > 0:
                prob = float(valid["survival_prob"].iloc[-1])
                rows.append({"group": group, "month": m, "survival_prob": round(prob, 4)})
            else:
                rows.append({"group": group, "month": m, "survival_prob": 1.0})
    return rows
