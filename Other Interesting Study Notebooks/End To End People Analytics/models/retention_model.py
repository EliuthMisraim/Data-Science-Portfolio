"""
retention_model.py
-------------------
Modelo de recomendación de intervención de retención.
Dado el perfil de un empleado, predice qué acción tiene mayor probabilidad
de reducir su riesgo de fuga.

Acciones posibles:
    - aumento_salarial
    - reducir_horas_extra
    - capacitacion
    - mentoria_liderazgo
    - reubicacion_departamento

Estrategia de entrenamiento:
    Se usan reglas de negocio expertas (heurísticas) para generar etiquetas sintéticas
    de entrenamiento a partir de los datos estructurados, y luego se entrena
    un RandomForestClassifier multi-clase.
"""

import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# Definición de las intervenciones posibles
INTERVENTIONS = [
    "aumento_salarial",
    "reducir_horas_extra",
    "capacitacion",
    "mentoria_liderazgo",
    "reubicacion_departamento",
]


def _assign_intervention(row: pd.Series) -> str:
    """
    Heurística de negocio para asignar la intervención más adecuada.
    Esta función genera las etiquetas sintéticas para el entrenamiento supervisado.
    """
    # 1. Horas extra excesivas → reducir horas
    if row.get("overtime_hours_last_month", 0) > 20:
        return "reducir_horas_extra"
    # 2. Carga de trabajo alta NOM-035 → reducir horas extra o mentoria
    if row.get("workload_index", 0) > 70:
        return "reducir_horas_extra"
    # 3. Liderazgo bajo → mentoría/reubicación
    if row.get("leadership_index", 50) < 40:
        return "mentoria_liderazgo"
    # 4. Salario bajo respecto al rol → aumento
    if row.get("salary_percentile", 50) < 30:
        return "aumento_salarial"
    # 5. Bajo desempeño en ventas → capacitación
    if row.get("quota_attainment_pct", 100) != -1 and row.get("quota_attainment_pct", 100) < 60:
        return "capacitacion"
    # 6. Bajo desempeño en almacén → capacitación
    if row.get("on_time_delivery_pct", 100) != -1 and row.get("on_time_delivery_pct", 100) < 75:
        return "capacitacion"
    # 7. Bradford Factor muy alto → reubicación (problema sistémico)
    if row.get("bradford_factor", 0) > 200:
        return "reubicacion_departamento"
    # Default: aumento salarial como acción más genérica de retención
    return "aumento_salarial"


def train_retention_model(db_path: str = "data/people_analytics.db", models_dir: str = "models"):
    """Entrena el modelo de recomendación de intervención y lo guarda como .pkl."""
    os.makedirs(models_dir, exist_ok=True)

    print("Cargando datos para el modelo de retención...")
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("""
        SELECT
            e.employee_id, e.role, e.age, e.gender, e.tenure_months,
            e.monthly_salary, e.overtime_hours_last_month, e.training_hours,
            e.last_performance_rating,
            n.workload_index, n.leadership_index, n.organizational_environment_index,
            s.quota_attainment_pct, s.commissions_earned,
            w.on_time_delivery_pct, w.inventory_errors
        FROM employees e
        LEFT JOIN nom035_survey n ON e.employee_id = n.employee_id
        LEFT JOIN performance_sales s ON e.employee_id = s.employee_id
        LEFT JOIN performance_warehouse w ON e.employee_id = w.employee_id
    """, conn)

    events_df = pd.read_sql_query(
        "SELECT employee_id, duration_days FROM absenteeism_events", conn
    )
    conn.close()

    # Bradford Factor
    bf = (
        events_df.groupby("employee_id")
        .agg(S=("duration_days", "count"), D=("duration_days", "sum"))
        .assign(bf=lambda x: x["S"] ** 2 * x["D"])["bf"]
    )
    df = df.join(bf.rename("bradford_factor"), on="employee_id")
    df["bradford_factor"] = df["bradford_factor"].fillna(0.0)

    # Percentil de salario por rol (para detectar si está mal pagado vs. peers)
    df["salary_percentile"] = df.groupby("role")["monthly_salary"].rank(pct=True) * 100

    # Rellenar NaN de métricas rol-específicas
    for col in ["quota_attainment_pct", "commissions_earned"]:
        df[col] = df[col].fillna(-1.0)
    for col in ["on_time_delivery_pct", "inventory_errors"]:
        df[col] = df[col].fillna(-1.0)

    # Generar etiquetas de intervención con la heurística experta
    df["intervention"] = df.apply(_assign_intervention, axis=1)
    print(f"Distribución de intervenciones:\n{df['intervention'].value_counts()}")

    # Encoding
    df["role_encoded"] = df["role"].map({"Ventas Técnicas": 1, "Almacén": 0})
    df["gender_encoded"] = df["gender"].map({"Masculino": 0, "Femenino": 1, "No binario": 2})

    feature_cols = [
        "role_encoded", "age", "gender_encoded", "tenure_months", "monthly_salary",
        "overtime_hours_last_month", "training_hours", "last_performance_rating",
        "workload_index", "leadership_index", "organizational_environment_index",
        "quota_attainment_pct", "commissions_earned",
        "on_time_delivery_pct", "inventory_errors",
        "bradford_factor", "salary_percentile",
    ]

    X = df[feature_cols]
    y = df["intervention"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    print("Entrenando RandomForest para recomendación de retención...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("\n--- Evaluación del Modelo de Retención ---")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Guardar modelo, encoder y feature names
    model_artifact = {
        "model": clf,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "interventions": INTERVENTIONS,
        "role_map": {"Ventas Técnicas": 1, "Almacén": 0},
        "gender_map": {"Masculino": 0, "Femenino": 1, "No binario": 2},
    }
    out_path = os.path.join(models_dir, "retention_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model_artifact, f)

    print(f"Modelo de retención guardado en: {out_path}")
    return clf, le


if __name__ == "__main__":
    train_retention_model()
