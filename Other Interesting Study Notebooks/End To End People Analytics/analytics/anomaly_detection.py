"""
anomaly_detection.py
---------------------
Detección de anomalías de ausentismo/comportamiento laboral usando IsolationForest.
Clasifica empleados como NORMAL, SOSPECHOSO o ANOMALÍA CRÍTICA.
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any


# Umbrales de clasificación basados en el Isolation Forest score
# (scores más negativos = más anómalos)
ANOMALY_LABELS = {
    "CRÍTICA": -0.15,    # Score < -0.15 → Anomalía crítica
    "SOSPECHOSO": 0.0,   # Score entre -0.15 y 0 → Sospechoso
    # Score > 0 → Normal
}


def _classify_score(score: float) -> str:
    if score < ANOMALY_LABELS["CRÍTICA"]:
        return "ANOMALÍA CRÍTICA"
    elif score < ANOMALY_LABELS["SOSPECHOSO"]:
        return "SOSPECHOSO"
    return "NORMAL"


def detect_anomalies(db_path: str, contamination: float = 0.1) -> Dict[str, Any]:
    """
    Detecta empleados con patrones atípicos de ausentismo/comportamiento.

    Features usadas:
    - bradford_factor: severidad del ausentismo (frecuencia² × días)
    - overtime_hours_last_month: horas extra
    - workload_index: carga de trabajo NOM-035
    - num_absence_events: número total de eventos de ausentismo
    - avg_absence_duration: duración promedio por evento

    Parámetros
    ----------
    db_path : str
        Ruta a la base de datos SQLite.
    contamination : float
        Proporción esperada de anomalías (0.0–0.5). Default = 0.1 (10%).

    Retorna
    -------
    Dict con:
        - anomalies: List[Dict] — todos los empleados con su clasificación y score
        - summary: conteo por categoría
        - feature_importance: importancia relativa de cada feature (basada en varianza)
    """
    conn = sqlite3.connect(db_path)
    try:
        # Cargar empleados activos con NOM-035
        df = pd.read_sql_query("""
            SELECT
                e.employee_id, e.name, e.role, e.department,
                e.overtime_hours_last_month, e.monthly_salary, e.tenure_months,
                n.workload_index, n.leadership_index
            FROM employees e
            LEFT JOIN nom035_survey n ON e.employee_id = n.employee_id
            WHERE e.has_quit = 0
        """, conn)

        # Calcular features de ausentismo por empleado
        events_df = pd.read_sql_query(
            "SELECT employee_id, duration_days FROM absenteeism_events", conn
        )
    finally:
        conn.close()

    # Unir eventos con empleados activos
    emp_ids = set(df["employee_id"].tolist())
    events_active = events_df[events_df["employee_id"].isin(emp_ids)]

    # Bradford Factor, nº eventos, duración promedio
    bf_stats = events_active.groupby("employee_id").agg(
        num_events=("duration_days", "count"),
        total_days=("duration_days", "sum"),
        avg_duration=("duration_days", "mean"),
    ).reset_index()
    bf_stats["bradford_factor"] = (bf_stats["num_events"] ** 2) * bf_stats["total_days"]

    df = df.merge(bf_stats, on="employee_id", how="left")
    df[["bradford_factor", "num_events", "total_days", "avg_duration"]] = (
        df[["bradford_factor", "num_events", "total_days", "avg_duration"]].fillna(0.0)
    )

    # Features para el modelo de detección
    feature_cols = [
        "bradford_factor",
        "overtime_hours_last_month",
        "workload_index",
        "num_events",
        "avg_duration",
    ]
    X = df[feature_cols].fillna(df[feature_cols].median())

    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar Isolation Forest
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_scaled)

    # Scores: más negativos = más anómalo
    scores = iso.score_samples(X_scaled)  # Puntuación de normalidad
    predictions = iso.predict(X_scaled)   # 1=normal, -1=anomalía

    df["anomaly_score"] = scores
    df["is_anomaly"] = predictions == -1
    df["anomaly_label"] = df["anomaly_score"].apply(_classify_score)

    # Calcular percentil de anomalía (0–100, donde 100 = más anómalo)
    min_s, max_s = scores.min(), scores.max()
    df["anomaly_percentile"] = ((df["anomaly_score"] - max_s) / (min_s - max_s + 1e-9) * 100).clip(0, 100)

    # Construir resultado
    anomalies = []
    for _, row in df.sort_values("anomaly_score").iterrows():
        anomalies.append({
            "employee_id": str(row["employee_id"]),
            "name": str(row["name"]),
            "role": str(row["role"]),
            "department": str(row["department"]),
            "anomaly_label": str(row["anomaly_label"]),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "anomaly_percentile": round(float(row["anomaly_percentile"]), 1),
            "bradford_factor": round(float(row["bradford_factor"]), 1),
            "num_absence_events": int(row["num_events"]),
            "avg_absence_duration_days": round(float(row["avg_duration"]), 1),
            "overtime_hours_last_month": round(float(row["overtime_hours_last_month"]), 1),
            "workload_index": round(float(row["workload_index"]), 1),
        })

    # Resumen de clasificación
    label_counts = df["anomaly_label"].value_counts().to_dict()
    summary = {
        "total_analyzed": int(len(df)),
        "normal": int(label_counts.get("NORMAL", 0)),
        "suspicious": int(label_counts.get("SOSPECHOSO", 0)),
        "critical": int(label_counts.get("ANOMALÍA CRÍTICA", 0)),
        "contamination_rate_pct": round(contamination * 100, 1),
    }

    # Importancia relativa de features (proxy: varianza de cada feature escalada)
    feature_variances = np.var(X_scaled, axis=0)
    total_var = feature_variances.sum()
    feature_importance = {
        col: round(float(v / total_var), 4)
        for col, v in zip(feature_cols, feature_variances)
    }

    return {
        "anomalies": anomalies,
        "summary": summary,
        "feature_importance": feature_importance,
    }
