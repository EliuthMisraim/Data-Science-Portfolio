-- Esquema de base de datos para People Analytics

-- Tabla principal de empleados
CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('Ventas Técnicas', 'Almacén')),
    department TEXT NOT NULL CHECK(department IN ('Ventas', 'Logística')),
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    tenure_months INTEGER NOT NULL,
    monthly_salary REAL NOT NULL,
    distance_km REAL NOT NULL,
    overtime_hours_last_month REAL NOT NULL,
    training_hours REAL NOT NULL,
    last_performance_rating INTEGER NOT NULL CHECK(last_performance_rating BETWEEN 1 AND 5),
    has_quit INTEGER NOT NULL CHECK(has_quit IN (0, 1)) DEFAULT 0
);

-- Tabla de desempeño de Ventas (CRM HubSpot)
CREATE TABLE IF NOT EXISTS performance_sales (
    employee_id TEXT PRIMARY KEY,
    quota_attainment_pct REAL NOT NULL,
    sales_closed INTEGER NOT NULL,
    commissions_earned REAL NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees (employee_id) ON DELETE CASCADE
);

-- Tabla de desempeño de Almacén/Logística (ERP)
CREATE TABLE IF NOT EXISTS performance_warehouse (
    employee_id TEXT PRIMARY KEY,
    on_time_delivery_pct REAL NOT NULL,
    inventory_errors INTEGER NOT NULL,
    machinery_incidents INTEGER NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees (employee_id) ON DELETE CASCADE
);

-- Tabla de resultados de la encuesta NOM-035 (Clima y Salud Organizacional)
CREATE TABLE IF NOT EXISTS nom035_survey (
    employee_id TEXT PRIMARY KEY,
    leadership_index REAL NOT NULL, -- Mayor es mejor (0-100)
    workload_index REAL NOT NULL,   -- Menor es mejor (0-100) (carga de trabajo/estrés)
    organizational_environment_index REAL NOT NULL, -- Mayor es mejor (0-100)
    FOREIGN KEY (employee_id) REFERENCES employees (employee_id) ON DELETE CASCADE
);

-- Tabla para el historial de eventos de ausentismo individual (para el cálculo del Bradford Factor)
CREATE TABLE IF NOT EXISTS absenteeism_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL,
    date TEXT NOT NULL, -- Formato YYYY-MM-DD
    duration_days INTEGER NOT NULL,
    reason TEXT NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees (employee_id) ON DELETE CASCADE
);

-- Tabla para el historial de series de tiempo agregadas de ausentismo (para Headcount Planning en Monterrey CDIX)
CREATE TABLE IF NOT EXISTS absenteeism_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start_date TEXT NOT NULL, -- Formato YYYY-MM-DD
    department TEXT NOT NULL,
    headcount INTEGER NOT NULL,
    hours_scheduled REAL NOT NULL,
    hours_absent REAL NOT NULL,
    absenteeism_rate REAL NOT NULL
);

-- Índices para optimizar consultas frecuentes
CREATE INDEX IF NOT EXISTS idx_employees_role ON employees(role);
CREATE INDEX IF NOT EXISTS idx_absenteeism_events_emp ON absenteeism_events(employee_id);
CREATE INDEX IF NOT EXISTS idx_absenteeism_history_date ON absenteeism_history(week_start_date);
