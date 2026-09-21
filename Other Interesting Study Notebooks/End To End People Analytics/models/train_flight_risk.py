"""
train_flight_risk.py
---------------------
Entrena el clasificador XGBoost de Flight Risk con:
  - Optimización bayesiana de hiperparámetros via Optuna (50 trials)
  - Tracking completo de experimentos via MLflow
  - Explicabilidad via SHAP TreeExplainer
"""

import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb
import shap

# MLflow — tracking de experimentos
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow no disponible, se omitirá el tracking de experimentos.")

# Optuna — optimización bayesiana de hiperparámetros
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna no disponible, se usarán hiperparámetros fijos.")


def load_data(db_path: str) -> tuple:
    """Carga y preprocesa los datos desde SQLite. Retorna (X, y, feature_cols, metadata)."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        e.employee_id, e.role, e.department, e.age, e.gender, e.tenure_months,
        e.monthly_salary, e.distance_km, e.overtime_hours_last_month, e.training_hours,
        e.last_performance_rating, e.has_quit,
        n.leadership_index, n.workload_index, n.organizational_environment_index,
        s.quota_attainment_pct, s.sales_closed, s.commissions_earned,
        w.on_time_delivery_pct, w.inventory_errors, w.machinery_incidents
    FROM employees e
    LEFT JOIN nom035_survey n ON e.employee_id = n.employee_id
    LEFT JOIN performance_sales s ON e.employee_id = s.employee_id
    LEFT JOIN performance_warehouse w ON e.employee_id = w.employee_id
    """
    df = pd.read_sql_query(query, conn)

    # Bradford Factor
    events_df = pd.read_sql_query("SELECT employee_id, date, duration_days FROM absenteeism_events", conn)
    conn.close()

    if not events_df.empty:
        bf_stats = events_df.groupby("employee_id").agg(
            S=("date", "count"), D=("duration_days", "sum")
        ).reset_index()
        bf_stats["bradford_factor"] = (bf_stats["S"] ** 2) * bf_stats["D"]
        bf_stats = bf_stats[["employee_id", "bradford_factor"]]
    else:
        bf_stats = pd.DataFrame(columns=["employee_id", "bradford_factor"])

    df = df.merge(bf_stats, on="employee_id", how="left")
    df["bradford_factor"] = df["bradford_factor"].fillna(0)

    # Rellenar NaN por rol
    fill_values = {
        "quota_attainment_pct": -1.0, "sales_closed": -1,
        "commissions_earned": -1.0, "on_time_delivery_pct": -1.0,
        "inventory_errors": -1, "machinery_incidents": -1
    }
    df = df.fillna(value=fill_values)

    # Encoding
    df["role_encoded"] = df["role"].map({"Ventas Técnicas": 1, "Almacén": 0})
    df["gender_encoded"] = df["gender"].map({"Masculino": 0, "Femenino": 1, "No binario": 2})

    feature_cols = [
        "role_encoded", "age", "gender_encoded", "tenure_months", "monthly_salary",
        "distance_km", "overtime_hours_last_month", "training_hours", "last_performance_rating",
        "leadership_index", "workload_index", "organizational_environment_index",
        "quota_attainment_pct", "sales_closed", "commissions_earned",
        "on_time_delivery_pct", "inventory_errors", "machinery_incidents",
        "bradford_factor"
    ]

    X = df[feature_cols]
    y = df["has_quit"]

    metadata = {
        "feature_cols": feature_cols,
        "role_map": {"Ventas Técnicas": 1, "Almacén": 0},
        "gender_map": {"Masculino": 0, "Femenino": 1, "No binario": 2}
    }

    return X, y, feature_cols, metadata


def train_model():
    db_path = "data/people_analytics.db"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(db_path):
        print(f"Error: No se encontró la base de datos en {db_path}. Corre los scripts de data/ primero.")
        return

    print("Conectando a la base de datos SQLite para extraer datos...")
    X, y, feature_cols, metadata = load_data(db_path)
    print(f"Dataset cargado con forma: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Muestra de entrenamiento: {X_train.shape[0]}, Muestra de test: {X_test.shape[0]}")

    scale_pos_weight = float((len(y_train) - sum(y_train)) / sum(y_train))

    # --- Configurar MLflow ---
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("people-analytics-flight-risk")
        print("MLflow tracking habilitado en http://127.0.0.1:5000")

    # --- Optimización con Optuna ---
    best_params = None

    if OPTUNA_AVAILABLE:
        print("\nIniciando optimización bayesiana de hiperparámetros con Optuna (50 trials)...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
                "eval_metric": "logloss",
                "use_label_encoder": False,
            }

            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

            # Loguear trial en MLflow si está disponible
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_roc_auc_mean", scores.mean())
                    mlflow.log_metric("cv_roc_auc_std", scores.std())

            return scores.mean()

        study = optuna.create_study(direction="maximize", study_name="xgboost-flight-risk")
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        best_params = study.best_params
        best_params["scale_pos_weight"] = scale_pos_weight
        best_params["random_state"] = 42
        best_params["eval_metric"] = "logloss"
        print(f"\nMejores hiperparámetros encontrados (ROC-AUC: {study.best_value:.4f}):")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        # Hiperparámetros fijos (fallback)
        best_params = {
            "n_estimators": 100, "max_depth": 4, "learning_rate": 0.08,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight, "random_state": 42,
            "eval_metric": "logloss"
        }

    # --- Entrenamiento final con los mejores parámetros ---
    print("\nEntrenando modelo final XGBoost con los mejores hiperparámetros...")
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n--- Resultados del Modelo Final en Test ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # --- Logging del modelo final en MLflow ---
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="flight_risk_final_model"):
            mlflow.log_params(best_params)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_roc_auc", roc_auc)
            mlflow.log_metric("train_size", X_train.shape[0])
            mlflow.log_metric("test_size", X_test.shape[0])
            mlflow.xgboost.log_model(model, "flight_risk_xgb_model")
            print("Modelo registrado en MLflow.")

    # --- Guardar artefactos ---
    with open(os.path.join(models_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print("Inicializando TreeExplainer de SHAP...")
    explainer = shap.TreeExplainer(model, data=X_train)

    with open(os.path.join(models_dir, "flight_risk_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(models_dir, "flight_risk_explainer.pkl"), "wb") as f:
        pickle.dump(explainer, f)

    X_train.to_csv(os.path.join(models_dir, "X_train.csv"), index=False)

    print(f"\nModelo guardado en: {models_dir}/flight_risk_model.pkl")
    print(f"Explicador SHAP guardado en: {models_dir}/flight_risk_explainer.pkl")
    print("Entrenamiento completado exitosamente.")


if __name__ == "__main__":
    train_model()
