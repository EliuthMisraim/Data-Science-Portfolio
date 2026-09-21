import os
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from api.schemas import (
    FlightRiskRequest, FlightRiskResponse, EmployeeRiskItem,
    AbsenteeismRequest, AbsenteeismResponse,
    MonteCarloRequest, MonteCarloResponse
)
from api.utils import ModelManager
from simulator.monte_carlo import MonteCarloSimulator

# Módulos analíticos v2.0
try:
    from analytics.cohort_analysis import get_cohort_analysis
    from analytics.survival_analysis import get_survival_curves
    from analytics.anomaly_detection import detect_anomalies
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    ANALYTICS_AVAILABLE = False
    print(f"ADVERTENCIA: Módulos de analytics no disponibles: {e}")

# Motor de alertas
try:
    from alerts.alert_engine import fire_alerts
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False
    print("ADVERTENCIA: Motor de alertas no disponible.")

# Inicializar aplicación FastAPI
app = FastAPI(
    title="People Analytics End-to-End API",
    description="Microservicios de Machine Learning, Deep Learning y Simulación de ROI Financiero para RRHH.",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde el frontend de Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar gestor de modelos y simulador
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'people_analytics.db')

model_manager = None
simulator = MonteCarloSimulator()

@app.on_event("startup")
def startup_event():
    global model_manager
    model_manager = ModelManager(models_dir=models_dir)
    print("API de People Analytics inicializada correctamente.")

@app.get("/")
def read_root():
    # Comprobar si los modelos están listos
    status = {
        "api_status": "online",
        "flight_risk_model_loaded": model_manager.flight_risk_model is not None if model_manager else False,
        "absenteeism_model_loaded": model_manager.absenteeism_model is not None if model_manager else False,
        "database_exists": os.path.exists(db_path)
    }
    return status

# --- Endpoint de Predicción de Flight Risk ---
@app.post("/predict/flight-risk", response_model=FlightRiskResponse)
def predict_flight_risk(request: FlightRiskRequest):
    if model_manager.flight_risk_model is None:
        raise HTTPException(status_code=503, detail="El modelo de Flight Risk no está cargado o entrenado.")
    try:
        prob, high_risk, shap_vals = model_manager.predict_flight_risk(request)
        return FlightRiskResponse(
            employee_id=request.employee_id,
            flight_risk_prob=prob,
            high_risk=high_risk,
            shap_values=shap_vals
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint de Predicción de Flight Risk por ID ---
@app.get("/predict/flight-risk/{employee_id}", response_model=FlightRiskResponse)
def predict_flight_risk_by_id(employee_id: str):
    if model_manager.flight_risk_model is None:
        raise HTTPException(status_code=503, detail="El modelo de Flight Risk no está cargado o entrenado.")
    
    conn = get_db_connection()
    try:
        # Cargar los datos de este empleado
        query = """
        SELECT 
            e.employee_id, e.role, e.department, e.age, e.gender, e.tenure_months, 
            e.monthly_salary, e.distance_km, e.overtime_hours_last_month, e.training_hours, 
            e.last_performance_rating,
            n.leadership_index, n.workload_index, n.organizational_environment_index,
            s.quota_attainment_pct, s.sales_closed, s.commissions_earned,
            w.on_time_delivery_pct, w.inventory_errors, w.machinery_incidents
        FROM employees e
        LEFT JOIN nom035_survey n ON e.employee_id = n.employee_id
        LEFT JOIN performance_sales s ON e.employee_id = s.employee_id
        LEFT JOIN performance_warehouse w ON e.employee_id = w.employee_id
        WHERE e.employee_id = ?
        """
        df = pd.read_sql_query(query, conn, params=(employee_id,))
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Empleado {employee_id} no encontrado.")
            
        # Calcular Bradford Factor
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(duration_days) FROM absenteeism_events WHERE employee_id = ?", (employee_id,))
        S, D = cur.fetchone()
        S = S or 0
        D = D or 0
        bf = (S ** 2) * D
        
        # Rellenar nulos
        row = df.iloc[0].to_dict()
        for k in ['quota_attainment_pct', 'sales_closed', 'commissions_earned', 
                  'on_time_delivery_pct', 'inventory_errors', 'machinery_incidents']:
            if pd.isna(row[k]) or row[k] is None:
                row[k] = -1.0 if 'pct' in k or 'earned' in k else -1
                
        # Crear objeto FlightRiskRequest
        req = FlightRiskRequest(
            employee_id=row['employee_id'],
            role=row['role'],
            age=int(row['age']),
            gender=row['gender'],
            tenure_months=int(row['tenure_months']),
            monthly_salary=float(row['monthly_salary']),
            distance_km=float(row['distance_km']),
            overtime_hours_last_month=float(row['overtime_hours_last_month']),
            training_hours=float(row['training_hours']),
            last_performance_rating=int(row['last_performance_rating']),
            leadership_index=float(row['leadership_index']),
            workload_index=float(row['workload_index']),
            organizational_environment_index=float(row['organizational_environment_index']),
            quota_attainment_pct=float(row['quota_attainment_pct']),
            sales_closed=int(row['sales_closed']),
            commissions_earned=float(row['commissions_earned']),
            on_time_delivery_pct=float(row['on_time_delivery_pct']),
            inventory_errors=int(row['inventory_errors']),
            machinery_incidents=int(row['machinery_incidents']),
            bradford_factor=float(bf)
        )
        
        prob, high_risk, shap_vals = model_manager.predict_flight_risk(req)
        return FlightRiskResponse(
            employee_id=employee_id,
            flight_risk_prob=prob,
            high_risk=high_risk,
            shap_values=shap_vals
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- Endpoint de Predicción de Ausentismo (LSTM) ---
@app.post("/predict/absenteeism", response_model=AbsenteeismResponse)
def predict_absenteeism(request: AbsenteeismRequest):
    if model_manager.absenteeism_model is None:
        raise HTTPException(status_code=503, detail="El modelo LSTM de ausentismo no está cargado o entrenado.")
    try:
        forecast = model_manager.forecast_absenteeism(request)
        return AbsenteeismResponse(
            department=request.department,
            forecasted_rates=forecast
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint de Simulación Monte Carlo (ROI) ---
@app.post("/simulate/roi", response_model=MonteCarloResponse)
def simulate_roi(request: MonteCarloRequest):
    try:
        # Convertir datos de la petición a formato de diccionario compatible con el simulador
        employees_list = [
            {'role': emp.role, 'monthly_salary': emp.monthly_salary, 'flight_risk_prob': emp.flight_risk_prob}
            for emp in request.employees
        ]
        
        sim_results = simulator.simulate_roi(
            employees_list=employees_list,
            global_intervention_cost=request.global_intervention_cost,
            effectiveness=request.effectiveness,
            n_simulations=request.n_simulations
        )
        
        return MonteCarloResponse(
            avg_savings=sim_results['avg_savings'],
            p5_savings=sim_results['p5_savings'],
            p95_savings=sim_results['p95_savings'],
            prob_positive_roi=sim_results['prob_positive_roi'],
            total_intervention_cost=sim_results['total_intervention_cost'],
            expected_roi_pct=sim_results['expected_roi_pct']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoints de Base de Datos para el Dashboard ---
def get_db_connection():
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Base de datos no encontrada. Ejecuta data/load_db.py primero.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/db/descriptive-stats")
def get_descriptive_stats():
    """
    Obtiene métricas descriptivas agregadas de la base de datos para el dashboard de salud organizacional.
    """
    conn = get_db_connection()
    try:
        # 1. KPIs Generales
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(CASE WHEN has_quit=1 THEN 1 ELSE 0 END) FROM employees")
        total, quit = cur.fetchone()
        
        # 2. Rotación por departamento
        cur.execute("""
            SELECT role, COUNT(*), SUM(CASE WHEN has_quit=1 THEN 1 ELSE 0 END)
            FROM employees GROUP BY role
        """)
        dept_turnover = []
        for r in cur.fetchall():
            dept_turnover.append({
                "role": r[0],
                "total": r[1],
                "quit": r[2],
                "turnover_rate": round((r[2] / r[1]) * 100, 1)
            })
            
        # 3. NOM-035 Promedios por Rol
        cur.execute("""
            SELECT e.role, 
                   AVG(n.leadership_index) as avg_leadership, 
                   AVG(n.workload_index) as avg_workload, 
                   AVG(n.organizational_environment_index) as avg_env
            FROM employees e
            JOIN nom035_survey n ON e.employee_id = n.employee_id
            GROUP BY e.role
        """)
        nom35_stats = []
        for r in cur.fetchall():
            nom35_stats.append({
                "role": r["role"],
                "avg_leadership": round(r["avg_leadership"], 1),
                "avg_workload": round(r["avg_workload"], 1),
                "avg_env": round(r["avg_env"], 1)
            })
            
        # 4. Datos de Desempeño promedio por Rol
        # Ventas
        cur.execute("SELECT AVG(quota_attainment_pct), AVG(commissions_earned) FROM performance_sales")
        sales_avg = cur.fetchone()
        
        # Almacén
        cur.execute("SELECT AVG(on_time_delivery_pct), AVG(inventory_errors), SUM(machinery_incidents) FROM performance_warehouse")
        wh_avg = cur.fetchone()
        
        perf_stats = {
            "sales": {
                "avg_quota_attainment": round(sales_avg[0], 1) if sales_avg[0] else 0.0,
                "avg_commissions": round(sales_avg[1], 2) if sales_avg[1] else 0.0
            },
            "warehouse": {
                "avg_on_time_delivery": round(wh_avg[0], 1) if wh_avg[0] else 0.0,
                "avg_inventory_errors": round(wh_avg[1], 2) if wh_avg[1] else 0.0,
                "total_machinery_incidents": wh_avg[2] if wh_avg[2] else 0
            }
        }
        
        return {
            "total_employees": total,
            "total_quit": quit,
            "general_turnover_rate": round((quit / total) * 100, 1) if total > 0 else 0.0,
            "dept_turnover": dept_turnover,
            "nom035_averages": nom35_stats,
            "performance_averages": perf_stats
        }
    finally:
        conn.close()

@app.get("/db/employees-risk", response_model=List[EmployeeRiskItem])
def get_employees_risk():
    """
    Lista todos los empleados activos, ejecuta el predictor XGBoost para cada uno,
    e integra sus datos de riesgo y Bradford Factor. Ordenado por riesgo de mayor a menor.
    """
    if model_manager.flight_risk_model is None:
        raise HTTPException(status_code=503, detail="El modelo de Flight Risk no está entrenado.")
        
    conn = get_db_connection()
    try:
        # Cargar los datos de los empleados activos (has_quit = 0)
        query = """
        SELECT 
            e.employee_id, e.name, e.role, e.department, e.age, e.gender, e.tenure_months, 
            e.monthly_salary, e.distance_km, e.overtime_hours_last_month, e.training_hours, 
            e.last_performance_rating,
            n.leadership_index, n.workload_index, n.organizational_environment_index,
            s.quota_attainment_pct, s.sales_closed, s.commissions_earned,
            w.on_time_delivery_pct, w.inventory_errors, w.machinery_incidents
        FROM employees e
        LEFT JOIN nom035_survey n ON e.employee_id = n.employee_id
        LEFT JOIN performance_sales s ON e.employee_id = s.employee_id
        LEFT JOIN performance_warehouse w ON e.employee_id = w.employee_id
        WHERE e.has_quit = 0
        """
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return []
            
        # Obtener Bradford Factor por empleado activo
        events_query = "SELECT employee_id, date, duration_days FROM absenteeism_events"
        df_events = pd.read_sql_query(events_query, conn)
        
        bf_dict = {}
        for emp_id in df['employee_id']:
            emp_evs = df_events[df_events['employee_id'] == emp_id]
            S = len(emp_evs)
            D = emp_evs['duration_days'].sum()
            bf_dict[emp_id] = (S ** 2) * D
            
        df['bradford_factor'] = df['employee_id'].map(bf_dict).fillna(0.0)
        
        # Rellenar nulos
        fill_values = {
            'quota_attainment_pct': -1.0,
            'sales_closed': -1,
            'commissions_earned': -1.0,
            'on_time_delivery_pct': -1.0,
            'inventory_errors': -1,
            'machinery_incidents': -1
        }
        df = df.fillna(value=fill_values)
        
        # Preparar entrada del modelo
        df['role_encoded'] = df['role'].map(model_manager.metadata['role_map'])
        df['gender_encoded'] = df['gender'].map(model_manager.metadata['gender_map'])
        
        X = df[model_manager.metadata['feature_cols']]
        
        # Predecir probabilidades
        probs = model_manager.flight_risk_model.predict_proba(X)[:, 1]
        df['flight_risk_prob'] = probs
        df['high_risk'] = probs >= 0.50
        
        # Seleccionar y ordenar resultado
        result_df = df[[
            'employee_id', 'name', 'role', 'monthly_salary', 'flight_risk_prob', 
            'high_risk', 'bradford_factor', 'workload_index', 'overtime_hours_last_month', 
            'last_performance_rating'
        ]].sort_values(by='flight_risk_prob', ascending=False)
        
        # Convertir a objetos del response_model
        res = []
        for idx, row in result_df.iterrows():
            res.append(EmployeeRiskItem(
                employee_id=row['employee_id'],
                name=row['name'],
                role=row['role'],
                monthly_salary=float(row['monthly_salary']),
                flight_risk_prob=float(row['flight_risk_prob']),
                high_risk=bool(row['high_risk']),
                bradford_factor=float(row['bradford_factor']),
                workload_index=float(row['workload_index']),
                overtime_hours_last_month=float(row['overtime_hours_last_month']),
                last_performance_rating=int(row['last_performance_rating'])
            ))

        # Disparar alertas automáticas si el motor está disponible
        if ALERTS_AVAILABLE:
            try:
                emp_list_for_alerts = [
                    {
                        'employee_id': r.employee_id,
                        'name': r.name,
                        'role': r.role,
                        'flight_risk_prob': r.flight_risk_prob,
                        'workload_index': r.workload_index,
                        'overtime_hours_last_month': r.overtime_hours_last_month,
                    }
                    for r in res
                ]
                alert_result = fire_alerts(emp_list_for_alerts)
                if alert_result['high_risk_count'] > 0:
                    print(
                        f"Motor de alertas: {alert_result['high_risk_count']} empleados en alto riesgo. "
                        f"Email enviado: {alert_result['email_sent']}, Slack enviado: {alert_result['slack_sent']}"
                    )
            except Exception as alert_err:
                print(f"Error en motor de alertas (no crtico): {alert_err}")

        return res
    finally:
        conn.close()

@app.get("/db/absenteeism-history/{dept}")
def get_absenteeism_history(dept: str):
    """
    Retorna el historial completo de ausentismo semanal de un departamento para gráficos.
    """
    if dept not in ['Almacén', 'Ventas Técnicas']:
        raise HTTPException(status_code=400, detail="Departamento inválido. Debe ser 'Almacén' o 'Ventas Técnicas'")
        
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT week_start_date, headcount, hours_scheduled, hours_absent, absenteeism_rate
            FROM absenteeism_history
            WHERE department = ?
            ORDER BY week_start_date ASC
        """, (dept,))
        
        res = []
        for r in cur.fetchall():
            res.append({
                "week_start_date": r[0],
                "headcount": r[1],
                "hours_scheduled": r[2],
                "hours_absent": r[3],
                "absenteeism_rate": r[4]
            })
        return res
    finally:
        conn.close()


# ============================================================
# ENDPOINTS ANALÍTICOS v2.0
# ============================================================

@app.get("/analytics/cohorts")
def get_cohorts():
    """
    Análisis de cohortes de empleados segmentados por antigüedad y departamento.
    Retorna tasa de rotación, Bradford Factor, NOM-035 e indicadores de riesgo por cohorte.
    """
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Módulo de analytics no disponible.")
    try:
        # Si el modelo está disponible, calcular probabilidades de riesgo para cohortes
        flight_risk_probs = None
        if model_manager and model_manager.flight_risk_model is not None:
            # Obtener todos los empleados y sus probabilidades precalculadas
            conn = get_db_connection()
            try:
                query = """
                SELECT e.employee_id, e.role, e.age, e.gender, e.tenure_months,
                       e.monthly_salary, e.distance_km, e.overtime_hours_last_month,
                       e.training_hours, e.last_performance_rating,
                       n.leadership_index, n.workload_index, n.organizational_environment_index,
                       s.quota_attainment_pct, s.sales_closed, s.commissions_earned,
                       w.on_time_delivery_pct, w.inventory_errors, w.machinery_incidents
                FROM employees e
                LEFT JOIN nom035_survey n ON e.employee_id = n.employee_id
                LEFT JOIN performance_sales s ON e.employee_id = s.employee_id
                LEFT JOIN performance_warehouse w ON e.employee_id = w.employee_id
                """
                df = pd.read_sql_query(query, conn)
                events_df = pd.read_sql_query(
                    "SELECT employee_id, duration_days FROM absenteeism_events", conn
                )
            finally:
                conn.close()

            # Bradford Factor
            bf = events_df.groupby('employee_id').agg(
                S=('duration_days', 'count'), D=('duration_days', 'sum')
            ).assign(bf=lambda x: x['S']**2 * x['D'])['bf']
            df = df.join(bf.rename('bradford_factor'), on='employee_id')
            df['bradford_factor'] = df['bradford_factor'].fillna(0.0)

            fill_vals = {
                'quota_attainment_pct': -1.0, 'sales_closed': -1,
                'commissions_earned': -1.0, 'on_time_delivery_pct': -1.0,
                'inventory_errors': -1, 'machinery_incidents': -1
            }
            df = df.fillna(value=fill_vals)
            df['role_encoded'] = df['role'].map(model_manager.metadata['role_map'])
            df['gender_encoded'] = df['gender'].map(model_manager.metadata['gender_map'])

            X = df[model_manager.metadata['feature_cols']]
            probs = model_manager.flight_risk_model.predict_proba(X)[:, 1]
            flight_risk_probs = dict(zip(df['employee_id'].tolist(), probs.tolist()))

        cohorts = get_cohort_analysis(db_path, flight_risk_probs=flight_risk_probs)
        return {"cohorts": cohorts, "total_cohorts": len(cohorts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/survival-curve")
def get_km_survival_curve():
    """
    Curvas de supervivencia Kaplan-Meier por grupo (Global, Almácn, Ventas Técnicas).
    Modela el tiempo hasta la renuncia como un evento de tiempo.
    """
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Módulo de analytics no disponible.")
    try:
        result = get_survival_curves(db_path)
        return result
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"lifelines no está instalado: {e}. Corre: pip install lifelines"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/anomalies")
def get_anomalies(contamination: float = 0.1):
    """
    Detección de anomalías de ausentismo/comportamiento con IsolationForest.
    Clasifica empleados activos como NORMAL, SOSPECHOSO o ANOMALÍA CRÍTICA.

    - contamination: Proporción esperada de anomalías (0.0–0.5). Default=0.1
    """
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Módulo de analytics no disponible.")
    if not (0.0 < contamination <= 0.5):
        raise HTTPException(
            status_code=422,
            detail="El parámetro contamination debe estar entre 0.01 y 0.5."
        )
    try:
        result = detect_anomalies(db_path, contamination=contamination)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/retention/{employee_id}")
def predict_retention_action(employee_id: str):
    """
    Predice la intervención de retención más efectiva para un empleado específico.
    Retorna la acción recomendada y las probabilidades de cada intervención posible.
    """
    if model_manager is None or model_manager.retention_model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo de retención no está cargado. Ejecuta models/retention_model.py."
        )

    conn = get_db_connection()
    try:
        query = """
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
        WHERE e.employee_id = ?
        """
        df = pd.read_sql_query(query, conn, params=(employee_id,))
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Empleado {employee_id} no encontrado.")

        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*), SUM(duration_days) FROM absenteeism_events WHERE employee_id = ?",
            (employee_id,)
        )
        S, D = cur.fetchone()
        S = S or 0; D = D or 0
        bf = (S ** 2) * D

        row = df.iloc[0].to_dict()
        row['bradford_factor'] = float(bf)
        for k in ['quota_attainment_pct', 'commissions_earned', 'on_time_delivery_pct', 'inventory_errors']:
            if pd.isna(row.get(k)):
                row[k] = -1.0

        # Encoding de variables categóricas
        row['role_encoded'] = model_manager.metadata['role_map'].get(row.get('role', ''), 0)
        row['gender_encoded'] = model_manager.metadata['gender_map'].get(row.get('gender', ''), 0)

        result = model_manager.predict_retention(row)
        result['employee_id'] = employee_id
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
