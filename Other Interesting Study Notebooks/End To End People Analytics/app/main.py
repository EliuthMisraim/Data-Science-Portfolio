import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página (Debe ser la primera llamada de Streamlit)
st.set_page_config(
    page_title="Suite de People Analytics End-to-End",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar componentes de visualización locales
try:
    from app.components import (
        plot_nom035_comparison,
        plot_absenteeism_history,
        plot_shap_contributions,
        plot_monte_carlo_distribution,
        plot_roi_gauge,
        plot_cohort_heatmap,
        plot_cohort_bars,
        plot_survival_curves,
        plot_anomaly_scatter,
        plot_retention_radar,
    )
except ModuleNotFoundError:
    # Si se ejecuta desde otra ruta, fallback a importación directa
    from components import (
        plot_nom035_comparison,
        plot_absenteeism_history,
        plot_shap_contributions,
        plot_monte_carlo_distribution,
        plot_roi_gauge,
        plot_cohort_heatmap,
        plot_cohort_bars,
        plot_survival_curves,
        plot_anomaly_scatter,
        plot_retention_radar,
    )

# Definir URL del Backend FastAPI
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Inyectar estilos CSS personalizados
styles_path = os.path.join(os.path.dirname(__file__), 'styles.css')
if os.path.exists(styles_path):
    with open(styles_path, 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    # Intento de cargar en rutas alternativas
    alt_path = 'app/styles.css'
    if os.path.exists(alt_path):
        with open(alt_path, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Helper para llamadas HTTP seguras
def fetch_api(endpoint, method="GET", json_data=None):
    try:
        url = f"{BACKEND_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=json_data, timeout=30)
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error del Servidor ({response.status_code}): {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ No se pudo conectar al servidor FastAPI en {BACKEND_URL}. Por favor, asegúrate de que el backend esté corriendo.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {str(e)}")
        return None

# --- ESTRUCTURA PRINCIPAL DE LA APP ---
st.markdown("<h1 class='main-title'>Suite de People Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ingeniería de Datos, Machine Learning y Deep Learning Aplicados a la Estrategia de Recursos Humanos</p>", unsafe_allow_html=True)

# Verificar salud de la API en la barra lateral
api_health = fetch_api("/")
with st.sidebar:
    st.markdown("### 🔌 Estado del Sistema")
    if api_health:
        st.success("Backend: Conectado (Online)")
        if api_health.get("flight_risk_model_loaded"):
            st.success("Modelo Flight Risk: Listo")
        else:
            st.warning("Modelo Flight Risk: No Cargado")
            
        if api_health.get("absenteeism_model_loaded"):
            st.success("Modelo LSTM: Listo")
        else:
            st.warning("Modelo LSTM: No Cargado")
            
        if api_health.get("database_exists"):
            st.success("Base de Datos SQL: Lista")
        else:
            st.error("Base de Datos SQL: No Encontrada")
    else:
        st.error("Backend: Desconectado (Offline)")
        st.info("Inicia la API corriendo: uvicorn api.main:app --reload")

    st.markdown("---")
    st.markdown("### 💡 Resumen del Negocio")
    st.write("Esta plataforma integra datos laborales con desempeño de negocio para predecir fugas, ausentismo y calcular el ROI de retención.")

# Crear las pestañas ejecutivas
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Dashboard de Salud Organizacional",
    "🚨 Consola de Alertas Predictivas",
    "💰 Simulador de Decisiones y ROI",
    "🔬 Analíticas Avanzadas",
])

# --- PESTAÑA 1: DASHBOARD DE SALUD ORGANIZACIONAL (DESCRIPTIVO) ---
with tab1:
    st.markdown("### Diagnóstico Organizacional Actual")
    
    # Obtener estadísticas de la API
    stats = fetch_api("/db/descriptive-stats")
    
    if stats:
        # Fila de KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Headcount Activo</div>
                <div class="metric-value">{stats['total_employees'] - stats['total_quit']}</div>
                <div class="metric-delta delta-positive">Activos en plantilla</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Bajas Acumuladas</div>
                <div class="metric-value">{stats['total_quit']}</div>
                <div class="metric-delta delta-negative">Histórico 24 meses</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Tasa de Rotación</div>
                <div class="metric-value">{stats['general_turnover_rate']}%</div>
                <div class="metric-delta delta-negative">Tasa acumulada global</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Para el cuarto KPI calcularemos el promedio del Bradford Factor consultando la lista de empleados
        active_emp_risk = fetch_api("/db/employees-risk")
        avg_bf = 0.0
        if active_emp_risk:
            avg_bf = np.mean([e['bradford_factor'] for e in active_emp_risk])
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Índice Bradford (Promedio)</div>
                <div class="metric-value">{avg_bf:.1f}</div>
                <div class="metric-delta {'delta-negative' if avg_bf > 125 else 'delta-positive'}">
                    {'Alerta: Ausentismo alto' if avg_bf > 125 else 'Severidad normal'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Sección de Gráficos de Clima NOM-035 y Desempeño
        st.markdown("#### Clima Laboral y NOM-035 por Departamento")
        col_chart1, col_chart2 = st.columns([3, 2])
        
        with col_chart1:
            # Gráfico de barras NOM-035
            fig_nom = plot_nom035_comparison(stats['nom035_averages'])
            st.plotly_chart(fig_nom, use_container_width=True)
            
        with col_chart2:
            # Distribución de rotación por rol
            dept_df = pd.DataFrame(stats['dept_turnover'])
            fig_pie = px.pie(
                dept_df, 
                values='total', 
                names='role', 
                title='Distribución de Plantilla por Rol de Negocio',
                color_discrete_sequence=['#4E79A7', '#E15759'],
                hole=0.4
            )
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Sección de ausentismo con pronóstico LSTM
        st.markdown("#### Planificación de Plantilla y Pronóstico de Ausentismo (Monterrey CDIX)")
        col_dept_select, col_forecast_btn = st.columns([3, 1])
        
        with col_dept_select:
            dept_selected = st.selectbox(
                "Selecciona el departamento para analizar ausentismo:",
                ["Almacén", "Ventas Técnicas"],
                key="absenteeism_dept_select"
            )
            
        # Cargar historial del departamento
        history_dept = fetch_api(f"/db/absenteeism-history/{dept_selected}")
        
        if history_dept:
            # Botón para generar el forecast LSTM
            forecast_rates = None
            
            with col_forecast_btn:
                st.write("") # Espaciador
                st.write("")
                run_forecast = st.button("🔮 Pronosticar con LSTM")
                
            if run_forecast:
                # Extraer las últimas 8 semanas de la base de datos para la API
                hist_rates = [h['absenteeism_rate'] for h in history_dept[-8:]]
                
                # Derivar la semana del año del último registro + 1
                last_date_str = history_dept[-1]['week_start_date']
                last_date = pd.to_datetime(last_date_str)
                target_wk = int((last_date + pd.Timedelta(weeks=1)).week)
                if target_wk == 0:
                    target_wk = 1
                
                # Llamar API LSTM
                forecast_res = fetch_api("/predict/absenteeism", method="POST", json_data={
                    "department": dept_selected,
                    "historical_rates": hist_rates,
                    "target_week_of_year": target_wk
                })
                
                if forecast_res:
                    forecast_rates = forecast_res['forecasted_rates']
                    st.toast("¡Pronóstico LSTM generado exitosamente!")
            
            # Graficar historial y pronóstico
            # Cargar el histórico del otro departamento para comparación de fondo
            other_dept = "Ventas Técnicas" if dept_selected == "Almacén" else "Almacén"
            history_other = fetch_api(f"/db/absenteeism-history/{other_dept}")
            
            # Graficar
            if dept_selected == "Almacén":
                fig_abs = plot_absenteeism_history(history_dept, history_other, forecast_almacen=forecast_rates)
            else:
                fig_abs = plot_absenteeism_history(history_other, history_dept, forecast_ventas=forecast_rates)
                
            st.plotly_chart(fig_abs, use_container_width=True)
            
            if forecast_rates:
                st.markdown(f"**Tasas de ausentismo pronosticadas para {dept_selected} (Semanas 1 a 4):**")
                cols = st.columns(4)
                for idx, val in enumerate(forecast_rates):
                    cols[idx].metric(f"Semana t+{idx+1}", f"{val*100:.2f}%")
    else:
        st.warning("No se pudieron cargar las estadísticas generales de la base de datos.")

# --- PESTAÑA 2: CONSOLA DE ALERTAS PREDICTIVAS (PRESCRIPTIVO - FLIGHT RISK & SHAP) ---
with tab2:
    st.markdown("### Alertas de Flight Risk (Próximos 90 Días)")
    st.markdown("Filtrado prioritario de colaboradores en riesgo potencial y desglose de motivos mediante XAI.")
    
    # Cargar todos los empleados activos con su riesgo precalculado por la API
    active_emp = fetch_api("/db/employees-risk")
    
    if active_emp:
        df_active = pd.DataFrame([e for e in active_emp])
        
        # Filtros interactivos
        col_fil1, col_fil2 = st.columns(2)
        with col_fil1:
            role_filter = st.selectbox(
                "Filtrar por Departamento/Rol:",
                ["Todos", "Almacén", "Ventas Técnicas"]
            )
        with col_fil2:
            risk_filter = st.selectbox(
                "Filtrar por Nivel de Riesgo:",
                ["Todos", "Alto Riesgo (>= 50%)", "Riesgo Medio (30% - 50%)", "Bajo Riesgo (< 30%)"]
            )
            
        # Aplicar filtros al DataFrame
        df_filtered = df_active.copy()
        
        if role_filter != "Todos":
            df_filtered = df_filtered[df_filtered['role'] == role_filter]
            
        if risk_filter == "Alto Riesgo (>= 50%)":
            df_filtered = df_filtered[df_filtered['flight_risk_prob'] >= 0.50]
        elif risk_filter == "Riesgo Medio (30% - 50%)":
            df_filtered = df_filtered[(df_filtered['flight_risk_prob'] >= 0.30) & (df_filtered['flight_risk_prob'] < 0.50)]
        elif risk_filter == "Bajo Riesgo (< 30%)":
            df_filtered = df_filtered[df_filtered['flight_risk_prob'] < 0.30]
            
        # Formatear la visualización
        df_display = df_filtered.copy()
        df_display['flight_risk_prob'] = df_display['flight_risk_prob'].apply(lambda x: f"{x*100:.1f}%")
        df_display['monthly_salary'] = df_display['monthly_salary'].apply(lambda x: f"${x:,.2f} MXN")
        df_display['bradford_factor'] = df_display['bradford_factor'].apply(lambda x: f"{x:.0f}")
        df_display['workload_index'] = df_display['workload_index'].apply(lambda x: f"{x:.1f}")
        
        df_display.columns = [
            'ID Empleado', 'Nombre Completo', 'Rol', 'Salario Mensual', 
            'Probabilidad Fuga', 'Es Alto Riesgo', 'Índice Bradford', 
            'Carga Trabajo (NOM)', 'Horas Extra', 'Evaluación Desempeño'
        ]
        
        # Mostrar tabla interactiva
        st.write(f"Mostrando {len(df_filtered)} colaboradores:")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Selección de empleado individual para análisis XAI SHAP
        st.markdown("---")
        st.markdown("### 🔍 Análisis de Explicabilidad y Retención Individual")
        
        emp_select_list = [f"{r['employee_id']} - {r['name']}" for idx, r in df_filtered.iterrows()]
        
        if emp_select_list:
            selected_emp = st.selectbox(
                "Selecciona un empleado para diagnosticar el porqué de su riesgo:",
                emp_select_list
            )
            
            selected_id = selected_emp.split(" - ")[0]
            
            # Llamar endpoint de predicción por ID para obtener probabilidad y valores SHAP
            prediction_res = fetch_api(f"/predict/flight-risk/{selected_id}")
            
            if prediction_res:
                prob = prediction_res['flight_risk_prob']
                shap_vals = prediction_res['shap_values']
                
                # Obtener detalles descriptivos del empleado seleccionado
                emp_details = df_filtered[df_filtered['employee_id'] == selected_id].iloc[0]
                
                col_det1, col_det2 = st.columns([1, 2])
                
                with col_det1:
                    st.markdown("#### Perfil del Empleado")
                    
                    # Badge de riesgo
                    badge_class = "badge-high" if prob >= 0.50 else ("badge-medium" if prob >= 0.30 else "badge-low")
                    badge_label = "Alto Riesgo" if prob >= 0.50 else ("Riesgo Medio" if prob >= 0.30 else "Bajo Riesgo")
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 20px;">
                        <span class="badge {badge_class}" style="font-size: 1.2rem; padding: 10px 20px;">
                            {badge_label} ({prob*100:.1f}%)
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detalles en formato lista
                    st.markdown(f"**Nombre:** {emp_details['name']}")
                    st.markdown(f"**ID:** {emp_details['employee_id']}")
                    st.markdown(f"**Rol:** {emp_details['role']}")
                    st.markdown(f"**Salario Base:** ${emp_details['monthly_salary']:,.2f} MXN")
                    st.markdown(f"**Evaluación Desempeño:** {emp_details['last_performance_rating']}/5")
                    st.markdown(f"**Horas Extra (Mes):** {emp_details['overtime_hours_last_month']:.1f} hrs")
                    st.markdown(f"**Índice de Bradford:** {emp_details['bradford_factor']:.0f}")
                    st.markdown(f"**Carga de Trabajo (NOM-035):** {emp_details['workload_index']:.1f}/100")
                    
                with col_det2:
                    # Graficar SHAP
                    fig_shap = plot_shap_contributions(shap_vals)
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                # Recomendaciones de Retención Automatizadas
                st.markdown("#### 💡 Recomendaciones Prescriptivas Automáticas")
                
                # Encontrar las 2 características que más empujan al riesgo (valores positivos más altos)
                sorted_shap = sorted(shap_vals.items(), key=lambda item: item[1], reverse=True)
                top_factors = [item[0] for item in sorted_shap if item[1] > 0][:2]
                
                if top_factors:
                    st.markdown("Basado en el modelo explicativo SHAP, los factores que más aumentan el riesgo de fuga son:")
                    
                    for factor in top_factors:
                        # Reglas del sistema experto prescriptivo
                        if "Horas Extra" in factor:
                            st.markdown(f"""
                            <div class="rec-container">
                                <div class="rec-title">⏳ Exceso de Horas Extra detectado</div>
                                <div class="rec-body">
                                    El colaborador acumula {emp_details['overtime_hours_last_month']:.1f} horas extra este mes. 
                                    <b>Sugerencia:</b> Reducir horas extra mediante una mejor planificación de turnos o contratación temporal. 
                                    El burnout por carga horaria es el principal predictor en su expediente.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif "Carga de Trabajo" in factor:
                            st.markdown(f"""
                            <div class="rec-container">
                                <div class="rec-title">🤯 Carga de Trabajo Crítica (NOM-035)</div>
                                <div class="rec-body">
                                    El índice de carga de trabajo es de {emp_details['workload_index']:.1f}. 
                                    <b>Sugerencia:</b> Redistribuir responsabilidades de tareas o programar una entrevista de 
                                    bienestar para balancear los entregables de logística/ventas.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif "Liderazgo" in factor:
                            st.markdown(f"""
                            <div class="rec-container">
                                <div class="rec-title">👥 Relación de Liderazgo Deficiente</div>
                                <div class="rec-body">
                                    El índice de percepción de liderazgo es bajo.
                                    <b>Sugerencia:</b> Se recomienda facilitar una mediación con su supervisor inmediato, realizar 
                                    una evaluación 360 del departamento, o reubicar al colaborador si persiste el conflicto.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif "Salario" in factor:
                            st.markdown(f"""
                            <div class="rec-container">
                                <div class="rec-title">💵 Descontento Salarial o Desviación del Tabulador</div>
                                <div class="rec-body">
                                    El salario base de ${emp_details['monthly_salary']:,.2f} MXN se percibe como insuficiente frente a su carga.
                                    <b>Sugerencia:</b> Realizar un ajuste salarial selectivo o un bono por cumplimiento de metas/comisiones.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif "Bradford" in factor:
                            st.markdown(f"""
                            <div class="rec-container">
                                <div class="rec-title">🤒 Ausentismo Frecuente (Índice de Bradford Alto)</div>
                                <div class="rec-body">
                                    El índice de Bradford de {emp_details['bradford_factor']:.0f} denota faltas repetitivas.
                                    <b>Sugerencia:</b> Programar una sesión de revisión para validar temas de salud personal o familiar. 
                                    El ausentismo reiterado está precediendo la deserción voluntaria.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif "Cuota" in factor or "Comisiones" in factor:
                            st.markdown(f"""
                            <div class="rec-container">
                                <div class="rec-title">🎯 Bajo Cumplimiento de Metas y Comisiones</div>
                                <div class="rec-body">
                                    Dificultades en HubSpot para el alcance de cuotas.
                                    <b>Sugerencia:</b> Proporcionar capacitación en ventas complejas, reducir la cuota temporalmente en la curva de aprendizaje, 
                                    o reestructurar el esquema de comisiones.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="rec-container">
                                <div class="rec-title">⚙️ Factor de Alerta: {factor}</div>
                                <div class="rec-body">
                                    Este factor tiene una contribución positiva en el flight risk.
                                    <b>Sugerencia:</b> Revisar detalladamente con el Business Partner de recursos humanos de su unidad.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.success("El empleado se encuentra en un estado de bajo riesgo y sus factores están balanceados.")

                # --- SECCIÓN: Recomendación de Retención (Modelo v2.0) ---
                st.markdown("---")
                st.markdown("#### Intervención de Retención Recomendada por el Modelo de IA")
                retention_res = fetch_api(f"/predict/retention/{selected_id}")
                if retention_res and "recommended_action" in retention_res:
                    action_labels = {
                        'aumento_salarial': 'Aumento Salarial',
                        'reducir_horas_extra': 'Reducir Horas Extra',
                        'capacitacion': 'Capacitación Profesional',
                        'mentoria_liderazgo': 'Mentoría / Mejora de Liderazgo',
                        'reubicacion_departamento': 'Reubicación de Departamento',
                    }
                    rec_action = retention_res['recommended_action']
                    rec_label = action_labels.get(rec_action, rec_action)
                    action_probs = retention_res.get('action_probabilities', {})

                    col_ret1, col_ret2 = st.columns([1, 2])
                    with col_ret1:
                        st.markdown(f"""
                        <div class="rec-container" style="border-left: 4px solid #4E79A7;">
                            <div class="rec-title" style="color:#4E79A7;">Intervención Óptima Recomendada</div>
                            <div class="rec-body" style="font-size:1.2rem;font-weight:bold;margin-top:8px;">
                                {rec_label}
                            </div>
                            <div class="rec-body" style="margin-top:8px;">
                                El modelo analizó el perfil completo del empleado y determinó que esta acción tiene
                                la mayor probabilidad de reducir el riesgo de fuga.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_ret2:
                        if action_probs:
                            fig_radar = plot_retention_radar(action_probs, rec_action)
                            st.plotly_chart(fig_radar, use_container_width=True)
                elif retention_res is None:
                    st.info("El modelo de retención no está disponible aún. Ejecuta models/retention_model.py.")
        else:
            st.info("Ningún colaborador coincide con los filtros aplicados.")
    else:
        st.warning("No se pudo cargar la lista de colaboradores activos.")

# --- PESTAÑA 3: SIMULADOR DE DECISIONES Y ROI (ESTRATÉGICO - MONTE CARLO) ---
with tab3:
    st.markdown("### Simulador de Decisiones Financieras de Retención")
    st.markdown("Diseña intervenciones de retención basadas en datos reales y evalúa el retorno de inversión bajo incertidumbre mediante simulaciones Monte Carlo.")
    
    if active_emp:
        # Cargar todos los empleados activos de la API
        df_active = pd.DataFrame([e for e in active_emp])
        
        # Parámetros del simulador en columnas
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            threshold_slider = st.slider(
                "🎯 Umbral de riesgo para intervención (%):",
                min_value=10, max_value=90, value=40, step=5,
                help="Aplicar retención solo a colaboradores con un Flight Risk mayor o igual a este valor."
            )
            
        with col_s2:
            cost_slider = st.slider(
                "💰 Costo de intervención por colaborador (MXN):",
                min_value=1000, max_value=25000, value=6000, step=500,
                help="Presupuesto a invertir por persona (e.g., bono de retención, aumento salarial trimestral, rediseño de turnos)."
            )
            
        with col_s3:
            effectiveness_slider = st.slider(
                "⚡ Efectividad esperada de la intervención (%):",
                min_value=10, max_value=100, value=50, step=5,
                help="¿En qué porcentaje se mitigará el riesgo de fuga? E.g., 50% significa que un riesgo de 80% pasará a ser 40%."
            )
            
        # Filtrar empleados elegibles para la simulación
        elig_mask = df_active['flight_risk_prob'] >= (threshold_slider / 100.0)
        df_elig = df_active[elig_mask]
        
        st.markdown("---")
        
        col_info1, col_info2 = st.columns([1, 2])
        
        with col_info1:
            st.markdown("#### Grupo Objetivo de Intervención")
            st.write(f"**Colaboradores en riesgo:** {len(df_elig)} de {len(df_active)} activos.")
            st.write(f"**Distribución por rol del grupo objetivo:**")
            st.write(df_elig['role'].value_counts())
            
            total_budget = len(df_elig) * cost_slider
            st.write(f"**Presupuesto Total de Inversión:** ${total_budget:,.2f} MXN")
            
            # Botón para ejecutar la simulación
            run_sim = st.button("🚀 Ejecutar Simulación Monte Carlo (5,000 corridas)")
            
        with col_info2:
            st.info("""
            **¿Cómo funciona la Simulación Monte Carlo?**
            - **Sin Intervención:** Cada colaborador en riesgo renuncia con base en la probabilidad calculada por XGBoost. Si se va, la empresa pierde dinero (costo estimado por rol de reclutamiento, onboarding y vacante comercial/logística).
            - **Con Intervención:** La empresa invierte el presupuesto de intervención en cada colaborador elegible, reduciendo su probabilidad de fuga según la efectividad. Si renuncia de todos modos, se incurren ambos costos (intervención + reemplazo).
            - **El ROI** se simula 5,000 veces modelando de forma probabilística la fluctuación del costo total y el ahorro neto de retención.
            """)
            
        # Si se hace click en simular
        if run_sim:
            if len(df_elig) == 0:
                st.error("No hay empleados elegibles para la simulación con el umbral seleccionado. Reduce el umbral de riesgo.")
            else:
                with st.spinner("Simulando escenarios financieros..."):
                    # Preparar la lista de empleados elegibles para mandar a la API
                    # La lista debe tener: 'role', 'monthly_salary', 'flight_risk_prob'
                    sim_payload = []
                    for idx, r in df_elig.iterrows():
                        sim_payload.append({
                            "role": r['role'],
                            "monthly_salary": float(r['monthly_salary']),
                            "flight_risk_prob": float(r['flight_risk_prob'])
                        })
                        
                    # Llamar API Monte Carlo
                    sim_res = fetch_api("/simulate/roi", method="POST", json_data={
                        "employees": sim_payload,
                        "global_intervention_cost": float(cost_slider),
                        "effectiveness": float(effectiveness_slider / 100.0),
                        "n_simulations": 5000
                    })
                    
                    if sim_res:
                        avg_sav = sim_res['avg_savings']
                        p5_sav = sim_res['p5_savings']
                        p95_sav = sim_res['p95_savings']
                        prob_pos = sim_res['prob_positive_roi']
                        expected_roi = sim_res['expected_roi_pct']
                        
                        # Generar el array de simulaciones para graficar a partir del promedio, p5 y p95
                        # Nota: la API no nos retorna el array gigante por desempeño, pero podemos
                        # simular localmente o generarlo rápidamente para graficar, o retornar un
                        # set resumido. Vamos a generar una distribución normal representativa localmente
                        # para graficar basándonos en P5, Promedio y P95 para fines visuales en Plotly!
                        # La distribución normal tiene media = avg_sav. Para simular la desviación estándar:
                        # P95 = mean + 1.645 * std => std = (p95 - mean) / 1.645
                        std_est = (p95_sav - avg_sav) / 1.645
                        sim_array = np.random.normal(avg_sav, std_est, 5000)
                        
                        st.markdown("### 📊 Resultados Proyectados de la Simulación")
                        
                        # KPIs de Simulación
                        col_r1, col_r2, col_r3 = st.columns(3)
                        with col_r1:
                            st.metric(
                                label="💰 Ahorro Neto Promedio Proyectado",
                                value=f"${avg_sav:,.2f} MXN",
                                delta=f"{expected_roi:.1f}% ROI esperado"
                            )
                        with col_r2:
                            st.metric(
                                label="🛡️ Probabilidad de Éxito ROI",
                                value=f"{prob_pos:.1f}%",
                                delta="ROI positivo esperado",
                                delta_color="normal"
                            )
                        with col_r3:
                            st.metric(
                                label="📉 Escenario Crítico (P5 - Peor caso)",
                                value=f"${p5_sav:,.2f} MXN",
                                delta="Con 95% de confianza"
                            )
                            
                        # Gráficos
                        col_g1, col_g2 = st.columns([2, 1])
                        with col_g1:
                            fig_mc = plot_monte_carlo_distribution(sim_array, p5_sav, p95_sav, avg_sav)
                            st.plotly_chart(fig_mc, use_container_width=True)
                        with col_g2:
                            fig_gauge = plot_roi_gauge(prob_pos)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                            
                        # Resumen estratégico
                        st.markdown("#### 🎯 Conclusión del Simulador para el Comité Ejecutivo")
                        if avg_sav > 0 and prob_pos > 80:
                            st.success(f"""
                            **RECOMENDACIÓN: APROBAR INTERVENCIÓN.**  
                            La simulación proyecta un ahorro financiero neto de **${avg_sav:,.2f} MXN** promedio, con una probabilidad del 
                            **{prob_pos:.1f}%** de obtener un retorno de inversión positivo. Incluso en el peor de los casos simulados (P5), 
                            el ahorro es de **${p5_sav:,.2f} MXN**, lo que significa que el riesgo financiero de ejecutar este presupuesto de retención es muy bajo.
                            """)
                        elif avg_sav > 0:
                            st.warning(f"""
                            **RECOMENDACIÓN: EVALUAR RIESGO.**  
                            El ahorro neto promedio proyectado es positivo (**${avg_sav:,.2f} MXN**), pero la probabilidad de retorno positivo es moderada 
                            (**{prob_pos:.1f}%**). Existe un riesgo no despreciable de incurrir en pérdidas debido a que algunos colaboradores de logística/ventas 
                            pueden renunciar a pesar de la intervención.
                            """)
                        else:
                            st.error(f"""
                            **RECOMENDACIÓN: RECHAZAR O REDISEÑAR INTERVENCIÓN.**  
                            El ahorro neto proyectado es negativo (**${avg_sav:,.2f} MXN**). El costo de implementar esta política supera el costo de reemplazo 
                            bajo la tasa de efectividad esperada. Se sugiere aumentar la efectividad de la retención o disminuir el costo de intervención por persona.
                            """)
    else:
        st.warning("No se pudo cargar la lista de colaboradores activos para simulación.")


# --- PESTAÑA 4: ANÁLISIS AVANZADO (COHORTES, KAPLAN-MEIER, ANOMALÍAS) ---
with tab4:
    st.markdown("### Análisis Avanzado de Capital Humano")
    st.markdown(
        "Modelos estadísticos avanzados para comprender la dinámica de retención: "
        "cohortes, tiempo de supervivencia y detección de patrones atípicos."
    )

    adv_tab1, adv_tab2, adv_tab3 = st.tabs([
        "📊 Análisis de Cohortes",
        "📉 Curvas de Supervivencia",
        "🚨 Detección de Anomalías"
    ])

    # Sub-pestaña 4.1: Análisis de Cohortes
    with adv_tab1:
        st.markdown("#### Segmentación por Cohorte de Antigüedad")
        st.markdown(
            "Compara el riesgo de rotación, carga de trabajo (NOM-035) e índice de Bradford "
            "entre cohortes de antigüedad para identificar los segmentos más vulnerables."
        )

        with st.spinner("Calculando cohortes..."):
            cohort_res = fetch_api("/analytics/cohorts")

        if cohort_res and "cohorts" in cohort_res:
            cohorts = cohort_res["cohorts"]
            df_cohorts = pd.DataFrame(cohorts)

            # Heatmap
            fig_heatmap = plot_cohort_heatmap(cohorts)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Bradford Factor bars
            fig_bf = plot_cohort_bars(cohorts)
            st.plotly_chart(fig_bf, use_container_width=True)

            # Tabla detallada
            st.markdown("#### Tabla Detallada de Cohortes")
            display_cols = [
                'cohort', 'role', 'headcount', 'active', 'quit',
                'turnover_rate_pct', 'avg_flight_risk_pct',
                'avg_bradford_factor', 'avg_workload_index', 'avg_overtime_hours'
            ]
            display_labels = [
                'Cohorte', 'Rol', 'Total', 'Activos', 'Bajas',
                'Rotación (%)', 'Riesgo Fuga (%)',
                'Bradford (Prom)', 'Carga Trabajo', 'Horas Extra'
            ]
            df_display = df_cohorts[display_cols].copy()
            df_display.columns = display_labels
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.warning("No se pudo cargar el análisis de cohortes.")

    # Sub-pestaña 4.2: Curvas de Supervivencia
    with adv_tab2:
        st.markdown("#### Análisis de Supervivencia Kaplan-Meier")
        st.markdown(
            "Modela el tiempo hasta la renuncia como un evento temporal. "
            "La curva muestra la probabilidad de que un empleado áun esté en la empresa "
            "a una antigüedad dada, con intervalos de confianza del 95%."
        )

        with st.spinner("Calculando curvas de supervivencia..."):
            survival_res = fetch_api("/analytics/survival-curve")

        if survival_res and "curves" in survival_res:
            fig_km = plot_survival_curves(survival_res)
            st.plotly_chart(fig_km, use_container_width=True)

            # Mostrar tabla de medianas de supervivencia
            if "median_survival" in survival_res:
                st.markdown("#### Mediana de Supervivencia por Grupo")
                median_data = [
                    {"Grupo": k, "Mediana (Meses)": f"{v:.1f}" if v else "No alcanzada"}
                    for k, v in survival_res["median_survival"].items()
                ]
                st.dataframe(pd.DataFrame(median_data), use_container_width=True, hide_index=True)

            # Resumen estadistico
            if "summary" in survival_res:
                summ = survival_res["summary"]
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Total Empleados Analizados", summ.get("total_employees", 0))
                col_s2.metric("Total Eventos (Renuncias)", summ.get("total_events", 0))
                col_s3.metric("Antigüedad Mediana (Activos)",
                              f"{summ.get('median_tenure_active_months', 0):.1f} meses")
        else:
            st.warning(
                "No se pudo cargar el análisis de supervivencia. "
                "Verifica que `lifelines` esté instalado: pip install lifelines"
            )

    # Sub-pestaña 4.3: Detección de Anomalías
    with adv_tab3:
        st.markdown("#### Detección de Anomalías de Ausentismo (Isolation Forest)")
        st.markdown(
            "Identifica empleados con patrones atípicos usando un modelo no supervisado. "
            "Considera bradford factor, horas extra, carga de trabajo y frecuencia de ausencias."
        )

        contamination_val = st.slider(
            "Proporción esperada de anomalías (%)",
            min_value=5, max_value=40, value=10, step=5,
            help="Porcentaje estimado de empleados con comportamiento atípico en la organización.",
            key="anomaly_contamination_slider"
        ) / 100.0

        with st.spinner("Ejecutando modelo de detección de anomalías..."):
            anomaly_res = fetch_api(f"/analytics/anomalies?contamination={contamination_val}")

        if anomaly_res and "anomalies" in anomaly_res:
            # Resumen de clasificación
            summ = anomaly_res["summary"]
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            col_a1.metric("Total Analizados", summ['total_analyzed'])
            col_a2.metric("Normales", summ['normal'], delta="Estado OK", delta_color="off")
            col_a3.metric(
                "Sospechosos", summ['suspicious'],
                delta="Revisar", delta_color="inverse"
            )
            col_a4.metric(
                "Anomalías Críticas", summ['critical'],
                delta="Acción Urgente", delta_color="inverse"
            )

            # Scatter plot
            fig_anomaly = plot_anomaly_scatter(anomaly_res["anomalies"])
            st.plotly_chart(fig_anomaly, use_container_width=True)

            # Tabla de anomalías críticas
            st.markdown("#### Empleados Sospechosos y Críticos (Requieren Acción)")
            df_anom = pd.DataFrame(anomaly_res["anomalies"])
            df_critical = df_anom[df_anom["anomaly_label"] != "NORMAL"].sort_values(
                "anomaly_percentile", ascending=False
            )
            if not df_critical.empty:
                disp_cols = [
                    'name', 'role', 'anomaly_label', 'anomaly_percentile',
                    'bradford_factor', 'overtime_hours_last_month', 'workload_index'
                ]
                disp_labels = [
                    'Nombre', 'Rol', 'Clasificación', 'Percentil Anomalía (%)',
                    'Bradford Factor', 'Horas Extra', 'Carga Trabajo'
                ]
                df_show = df_critical[disp_cols].copy()
                df_show.columns = disp_labels
                st.dataframe(df_show, use_container_width=True, hide_index=True)
            else:
                st.success("Ningún empleado fue clasificado como sospechoso o crítico.")
        else:
            st.warning("No se pudo cargar el análisis de anomalías.")

