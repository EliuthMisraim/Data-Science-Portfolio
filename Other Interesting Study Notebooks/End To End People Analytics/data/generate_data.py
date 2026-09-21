import os
import random
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Establecer semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

def generate_mock_data():
    print("Iniciando generación de Mock Data...")
    
    # Listas de nombres en español
    first_names = ['Juan', 'Carlos', 'Luis', 'María', 'Ana', 'José', 'Laura', 'Pedro', 'Sofía', 'Miguel', 
                   'Alejandro', 'Gabriela', 'Javier', 'Elena', 'Fernando', 'Patricia', 'Daniel', 'Rosa', 
                   'Roberto', 'Carmen', 'Eduardo', 'Guadalupe', 'Jorge', 'Leticia', 'Ricardo', 'Monica', 
                   'Arturo', 'Silvia', 'Francisco', 'Adriana']
    
    last_names = ['García', 'Rodríguez', 'González', 'Hernández', 'López', 'Martínez', 'Pérez', 'Sánchez', 
                  'Ramírez', 'Torres', 'Flores', 'Gómez', 'Díaz', 'Morales', 'Vázquez', 'Jiménez', 'Reyes', 
                  'Ruiz', 'Alvarez', 'Castillo', 'Moreno', 'Ortiz', 'Ramos', 'Herrera', 'Medina', 'Vargas', 
                  'Castro', 'Guzmán', 'Mendoza', 'Salazar']
    
    num_employees = 1000
    
    # 1. GENERACIÓN DE DATOS DEMOGRÁFICOS Y LABORALES BÁSICOS (employees)
    employees = []
    
    # Distribución de roles: 30% Ventas Técnicas, 70% Almacén (Logística)
    roles = ['Ventas Técnicas'] * 300 + ['Almacén'] * 700
    random.shuffle(roles)
    
    for i in range(num_employees):
        emp_id = f"EMP{i+1:04d}"
        role = roles[i]
        dept = 'Ventas' if role == 'Ventas Técnicas' else 'Logística'
        
        gender = np.random.choice(['Masculino', 'Femenino', 'No binario'], p=[0.55, 0.43, 0.02])
        
        # Asignar nombres coherentemente con género (aproximado)
        name = f"{random.choice(first_names)} {random.choice(last_names)} {random.choice(last_names)}"
        
        age = int(np.random.triangular(20, 32, 60))
        tenure_months = int(np.random.exponential(scale=30) + 1)
        tenure_months = min(tenure_months, (age - 18) * 12) # No puede haber trabajado antes de los 18
        
        # Salarios diferenciados por rol (pesos mexicanos MXN)
        if role == 'Ventas Técnicas':
            monthly_salary = round(float(np.random.normal(32000, 5000)), 2)
            monthly_salary = max(18000.0, min(monthly_salary, 55000.0))
            distance_km = round(float(np.random.gamma(shape=3, scale=5)), 1)
            overtime_hours = round(float(np.random.gamma(shape=1.5, scale=6)), 1)
        else: # Almacén
            monthly_salary = round(float(np.random.normal(11000, 1500)), 2)
            monthly_salary = max(8000.0, min(monthly_salary, 18000.0))
            distance_km = round(float(np.random.gamma(shape=4, scale=6)), 1)
            overtime_hours = round(float(np.random.gamma(shape=3.5, scale=8)), 1)
            
        distance_km = max(0.5, distance_km)
        overtime_hours = max(0.0, overtime_hours)
        
        training_hours = int(np.random.poisson(lam=12))
        last_perf = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.55, 0.20, 0.05]))
        
        employees.append({
            'employee_id': emp_id,
            'name': name,
            'role': role,
            'department': dept,
            'age': age,
            'gender': gender,
            'tenure_months': tenure_months,
            'monthly_salary': monthly_salary,
            'distance_km': distance_km,
            'overtime_hours_last_month': overtime_hours,
            'training_hours': training_hours,
            'last_performance_rating': last_perf
        })
        
    df_employees = pd.DataFrame(employees)
    
    # 2. GENERACIÓN DE CLIMA NOM-035 (nom035_survey)
    # Correlacionaremos de manera realista estos índices con la carga de trabajo y las horas extra
    nom035 = []
    for idx, row in df_employees.iterrows():
        emp_id = row['employee_id']
        role = row['role']
        overtime = row['overtime_hours_last_month']
        
        # Carga de trabajo (workload_index) relacionada con horas extra
        # A más horas extra, mayor carga de trabajo
        workload_base = 40 + (overtime * 1.5)
        workload = np.random.normal(workload_base, 8)
        workload = max(0.0, min(100.0, workload))
        
        # Liderazgo e índice organizacional
        # Si la carga es muy alta, el entorno organizacional tiende a verse afectado negativamente
        leadership = np.random.normal(70 - (workload / 10), 10)
        leadership = max(0.0, min(100.0, leadership))
        
        env_base = 75 - (workload / 8) + (leadership / 8)
        env = np.random.normal(env_base, 8)
        env = max(0.0, min(100.0, env))
        
        nom035.append({
            'employee_id': emp_id,
            'leadership_index': round(leadership, 1),
            'workload_index': round(workload, 1),
            'organizational_environment_index': round(env, 1)
        })
        
    df_nom035 = pd.DataFrame(nom035)
    
    # 3. GENERACIÓN DE DESEMPEÑO CRM Y ERP (performance_sales / performance_warehouse)
    sales_perf = []
    warehouse_perf = []
    
    for idx, row in df_employees.iterrows():
        emp_id = row['employee_id']
        role = row['role']
        perf_rating = row['last_performance_rating']
        
        if role == 'Ventas Técnicas':
            # Quota attainment vinculada a su evaluación de desempeño anterior
            quota_mean = 70 + (perf_rating * 8) # Ratings de 1-5 dan medias de 78 a 110%
            quota = np.random.normal(quota_mean, 12)
            quota = max(40.0, min(140.0, quota))
            
            sales_closed = int(np.random.poisson(lam=(quota / 4)))
            sales_closed = max(1, sales_closed)
            
            # Comisiones ganadas (proporcionales al cumplimiento de cuota y salario)
            commissions = 0.0
            if quota > 80:
                commissions = round(row['monthly_salary'] * (quota - 80) / 100 * 0.8, 2)
                
            sales_perf.append({
                'employee_id': emp_id,
                'quota_attainment_pct': round(quota, 1),
                'sales_closed': sales_closed,
                'commissions_earned': commissions
            })
            
        else: # Almacén
            # Entregas a tiempo e incidencias
            delivery_mean = 78 + (perf_rating * 4) # 82 a 98%
            delivery = np.random.normal(delivery_mean, 5)
            delivery = max(50.0, min(100.0, delivery))
            
            # Errores de inventario inversamente proporcionales al performance
            errors_lam = max(0.5, 6 - perf_rating)
            errors = np.random.poisson(lam=errors_lam)
            
            # Incidentes de maquinaria (raros, pero más frecuentes en bajo desempeño)
            incident_prob = 0.02 * (6 - perf_rating) # 2% a 10%
            incidents = 1 if random.random() < incident_prob else 0
            if incidents == 1 and random.random() < 0.15:
                incidents = 2 # Caso raro de 2 incidentes
                
            warehouse_perf.append({
                'employee_id': emp_id,
                'on_time_delivery_pct': round(delivery, 1),
                'inventory_errors': errors,
                'machinery_incidents': incidents
            })
            
    df_sales_perf = pd.DataFrame(sales_perf)
    df_warehouse_perf = pd.DataFrame(warehouse_perf)
    
    # 4. GENERACIÓN DE EVENTOS DE AUSENTISMO INDIVIDUAL (absenteeism_events)
    # Generaremos eventos de los últimos 12 meses
    absenteeism_events = []
    reasons = ['Enfermedad general', 'Trámite personal', 'Accidente menor', 'Falta injustificada', 'Permiso sindical']
    reason_p = [0.65, 0.15, 0.08, 0.10, 0.02]
    
    start_date = datetime.now() - timedelta(days=365)
    
    event_counter = 1
    for idx, row in df_employees.iterrows():
        emp_id = row['employee_id']
        role = row['role']
        workload = df_nom035.loc[df_nom035['employee_id'] == emp_id, 'workload_index'].values[0]
        env = df_nom035.loc[df_nom035['employee_id'] == emp_id, 'organizational_environment_index'].values[0]
        
        # Determinar propensión al ausentismo basada en rol, carga de trabajo y entorno organizacional
        # Los trabajadores de Almacén tienen mayor tasa de ausentismo promedio
        base_events = 2.0 if role == 'Almacén' else 0.8
        # Carga alta y mal entorno aumentan ausencias
        lambda_events = base_events + (workload / 25) - (env / 40)
        lambda_events = max(0.1, lambda_events)
        
        num_absences = np.random.poisson(lam=lambda_events)
        
        for _ in range(num_absences):
            days_offset = random.randint(1, 360)
            event_date = (start_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
            
            # Duración del ausentismo: corta (1-2 días) vs larga (3-5 días)
            # Faltas injustificadas y trámites son de 1 día típicamente. Enfermedades o accidentes son más largos
            reason = np.random.choice(reasons, p=reason_p)
            if reason in ['Trámite personal', 'Falta injustificada']:
                duration = 1
            elif reason == 'Accidente menor':
                duration = random.choice([2, 3, 4, 5])
            else: # Enfermedad o permiso
                duration = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.10, 0.05])
                
            absenteeism_events.append({
                'event_id': event_counter,
                'employee_id': emp_id,
                'date': event_date,
                'duration_days': int(duration),
                'reason': reason
            })
            event_counter += 1
            
    df_absenteeism_events = pd.DataFrame(absenteeism_events)
    
    # 5. CÁLCULO DEL TARGET 'has_quit' BASADO EN LOGIT REALISTA (Flight Risk)
    # Calculamos el Bradford Factor para usarlo en la decisión lógica
    # Bradford = S^2 * D (S = spells, D = total days)
    bf_dict = {}
    for emp_id in df_employees['employee_id']:
        emp_events = [e for e in absenteeism_events if e['employee_id'] == emp_id]
        S = len(emp_events)
        D = sum(e['duration_days'] for e in emp_events)
        bf_dict[emp_id] = S**2 * D
        
    has_quit_list = []
    for idx, row in df_employees.iterrows():
        emp_id = row['employee_id']
        role = row['role']
        overtime = row['overtime_hours_last_month']
        salary = row['monthly_salary']
        distance = row['distance_km']
        perf = row['last_performance_rating']
        
        survey = df_nom035[df_nom035['employee_id'] == emp_id].iloc[0]
        workload = survey['workload_index']
        leadership = survey['leadership_index']
        env = survey['organizational_environment_index']
        
        bf = bf_dict[emp_id]
        
        # Modelo Logístico de Riesgo
        # Ventas Técnicas vs Almacén
        if role == 'Almacén':
            # Salario relativo al promedio de Almacén
            salary_ratio = salary / 11000.0
            
            # Logit score
            logit = -3.8 \
                    + 0.07 * overtime \
                    + 0.04 * workload \
                    - 0.03 * leadership \
                    - 0.03 * env \
                    + 0.04 * distance \
                    - 0.40 * (perf - 3) \
                    + 0.001 * bf \
                    - 0.80 * salary_ratio
        else: # Ventas Técnicas
            sales = df_sales_perf[df_sales_perf['employee_id'] == emp_id].iloc[0]
            quota = sales['quota_attainment_pct']
            salary_ratio = salary / 32000.0
            
            logit = -4.2 \
                    + 0.03 * overtime \
                    - 0.05 * (quota - 90) \
                    - 0.02 * leadership \
                    - 0.03 * env \
                    + 0.05 * distance \
                    - 0.30 * (perf - 3) \
                    - 0.90 * salary_ratio
                    
        # Aplicar Sigmoide
        prob = 1.0 / (1.0 + math.exp(-logit))
        
        # Mapear probabilidad en un indicador binario (deserción en los próximos 90 días)
        # Hacemos que la tasa de rotación anual sea aproximadamente ~12-15%
        has_quit = 1 if random.random() < prob else 0
        has_quit_list.append(has_quit)
        
    df_employees['has_quit'] = has_quit_list
    
    # 6. HISTORIAL SEMANAL DE AUSENTISMO (Monterrey CDIX) - 104 semanas
    # Modelaremos un comportamiento de series temporales con tendencia, estacionalidad y picos
    history = []
    
    # Supongamos que en el CDIX de Monterrey hay 150 empleados en Almacén y 50 en Ventas
    base_headcounts = {'Almacén': 150, 'Ventas Técnicas': 50}
    
    # Empezar hace 104 semanas
    start_week = datetime.now() - timedelta(weeks=104)
    
    for w in range(104):
        week_date = (start_week + timedelta(weeks=w)).strftime('%Y-%m-%d')
        
        for dept_role in ['Almacén', 'Ventas Técnicas']:
            hc = base_headcounts[dept_role]
            # Pequeño crecimiento del headcount a lo largo del tiempo
            hc = int(hc * (1.0 + 0.0015 * w)) 
            
            # Horas programadas a la semana
            # Almacén: 48 horas semanales; Ventas: 40 horas semanales
            hours_per_emp = 48 if dept_role == 'Almacén' else 40
            hours_scheduled = hc * hours_per_emp
            
            # Modelo de Series Temporales para ausencias (T + S + H + E)
            # Tendencia
            trend = 0.0001 * w # Ligera tendencia al alza en ausentismo
            
            # Estacionalidad anual (pico en invierno - semana 50 a 4, valle en primavera - semana 15 a 22)
            # Usamos una función seno basada en el número de semana del año (0-51)
            week_of_year = (w % 52)
            seasonality = 0.015 * math.sin(2 * math.pi * (week_of_year - 38) / 52) 
            
            # Hitos / Eventos especiales (Picos de ausentismo por festividades)
            holiday_peak = 0.0
            if week_of_year in [50, 51, 0, 1]: # Navidad y Año Nuevo
                holiday_peak = 0.035
            elif week_of_year in [18, 19]: # Día de las Madres / Semana Santa (aprox)
                holiday_peak = 0.02
            elif week_of_year == 37: # Fiestas Patrias México (16 Septiembre)
                holiday_peak = 0.015
                
            # Tasa base de ausentismo (Almacén tiene más ausencias)
            base_rate = 0.045 if dept_role == 'Almacén' else 0.018
            
            # Ruido aleatorio
            noise = np.random.normal(0, 0.005)
            
            # Tasa final
            rate = base_rate + trend + seasonality + holiday_peak + noise
            rate = max(0.005, min(0.15, rate))
            
            hours_absent = round(hours_scheduled * rate, 1)
            final_rate = round(hours_absent / hours_scheduled, 4)
            
            history.append({
                'week_start_date': week_date,
                'department': dept_role,
                'headcount': hc,
                'hours_scheduled': float(hours_scheduled),
                'hours_absent': float(hours_absent),
                'absenteeism_rate': float(final_rate)
            })
            
    df_absenteeism_history = pd.DataFrame(history)
    
    print(f"Generados: {len(df_employees)} empleados.")
    print(f"Generados: {len(df_nom035)} registros NOM-035.")
    print(f"Generados: {len(df_sales_perf)} registros de ventas.")
    print(f"Generados: {len(df_warehouse_perf)} registros de almacén.")
    print(f"Generados: {len(df_absenteeism_events)} eventos de ausentismo.")
    print(f"Generados: {len(df_absenteeism_history)} registros históricos de series de tiempo.")
    
    # Crear carpeta para datos si no existe
    os.makedirs('data', exist_ok=True)
    
    # Guardar en CSV temporalmente para su posterior carga en SQL
    df_employees.to_csv('data/employees.csv', index=False)
    df_nom035.to_csv('data/nom035_survey.csv', index=False)
    df_sales_perf.to_csv('data/performance_sales.csv', index=False)
    df_warehouse_perf.to_csv('data/performance_warehouse.csv', index=False)
    df_absenteeism_events.to_csv('data/absenteeism_events.csv', index=False)
    df_absenteeism_history.to_csv('data/absenteeism_history.csv', index=False)
    print("Todos los archivos CSV fueron guardados en la carpeta /data.")
    
    return df_employees, df_nom035, df_sales_perf, df_warehouse_perf, df_absenteeism_events, df_absenteeism_history

if __name__ == '__main__':
    generate_mock_data()
