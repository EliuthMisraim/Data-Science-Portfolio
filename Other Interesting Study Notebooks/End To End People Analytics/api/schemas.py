from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- Flight Risk Schemas ---
class FlightRiskRequest(BaseModel):
    employee_id: Optional[str] = "EMP0000"
    role: str = Field(..., description="Ventas Técnicas o Almacén")
    age: int = Field(..., ge=18, le=100)
    gender: str = Field(..., description="Masculino, Femenino o No binario")
    tenure_months: int = Field(..., ge=0)
    monthly_salary: float = Field(..., ge=0)
    distance_km: float = Field(..., ge=0)
    overtime_hours_last_month: float = Field(..., ge=0)
    training_hours: float = Field(..., ge=0)
    last_performance_rating: int = Field(..., ge=1, le=5)
    leadership_index: float = Field(..., ge=0, le=100)
    workload_index: float = Field(..., ge=0, le=100)
    organizational_environment_index: float = Field(..., ge=0, le=100)
    quota_attainment_pct: Optional[float] = -1.0
    sales_closed: Optional[int] = -1
    commissions_earned: Optional[float] = -1.0
    on_time_delivery_pct: Optional[float] = -1.0
    inventory_errors: Optional[int] = -1
    machinery_incidents: Optional[int] = -1
    bradford_factor: Optional[float] = 0.0

class FlightRiskResponse(BaseModel):
    employee_id: str
    flight_risk_prob: float
    high_risk: bool
    shap_values: Dict[str, float]

class EmployeeRiskItem(BaseModel):
    employee_id: str
    name: str
    role: str
    monthly_salary: float
    flight_risk_prob: float
    high_risk: bool
    bradford_factor: float
    workload_index: float
    overtime_hours_last_month: float
    last_performance_rating: int

# --- Absenteeism Schemas ---
class AbsenteeismRequest(BaseModel):
    department: str = Field(..., description="Almacén o Ventas Técnicas")
    historical_rates: List[float] = Field(..., description="Tasa de ausentismo de las últimas 8 semanas", min_items=8, max_items=8)
    target_week_of_year: int = Field(..., description="Número de semana en el año para el inicio de la predicción (1-52)", ge=1, le=52)

class AbsenteeismResponse(BaseModel):
    department: str
    forecasted_rates: List[float] = Field(..., description="Tasas de ausentismo pronosticadas para las siguientes 4 semanas")

# --- Monte Carlo ROI Schemas ---
class MonteCarloEmployeeItem(BaseModel):
    role: str
    monthly_salary: float
    flight_risk_prob: float

class MonteCarloRequest(BaseModel):
    employees: List[MonteCarloEmployeeItem]
    global_intervention_cost: float = Field(..., description="Costo de intervención por empleado en MXN", ge=0)
    effectiveness: float = Field(..., description="Efectividad de retención (0.0 a 1.0)", ge=0.0, le=1.0)
    n_simulations: Optional[int] = Field(5000, description="Número de iteraciones", ge=100)

class MonteCarloResponse(BaseModel):
    avg_savings: float
    p5_savings: float
    p95_savings: float
    prob_positive_roi: float
    total_intervention_cost: float
    expected_roi_pct: float
    # Opcional: retornar los ahorros percentiles para graficar o mostrar
