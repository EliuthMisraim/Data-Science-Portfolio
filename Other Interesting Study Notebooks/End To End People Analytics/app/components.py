import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_nom035_comparison(nom035_averages):
    """
    Grafica la comparación de las métricas NOM-035 por rol.
    """
    df = pd.DataFrame(nom035_averages)
    
    # Derivar nombres legibles
    fig = go.Figure()
    
    # Liderazgo
    fig.add_trace(go.Bar(
        x=df['role'],
        y=df['avg_leadership'],
        name='Índice de Liderazgo (Líderes)',
        marker_color='#4E79A7'
    ))
    
    # Carga de Trabajo
    fig.add_trace(go.Bar(
        x=df['role'],
        y=df['avg_workload'],
        name='Carga de Trabajo (Estrés)',
        marker_color='#E15759'
    ))
    
    # Entorno Organizacional
    fig.add_trace(go.Bar(
        x=df['role'],
        y=df['avg_env'],
        name='Entorno Organizacional',
        marker_color='#59A14F'
    ))
    
    fig.update_layout(
        title=' NOM-035: Factores de Riesgo Psicosocial por Departamento',
        xaxis_title='Rol / Departamento',
        yaxis_title='Puntuación Promedio (0-100)',
        barmode='group',
        legend_title='Variables NOM-035',
        template='plotly_white',
        height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_absenteeism_history(history_almacen, history_ventas, forecast_almacen=None, forecast_ventas=None):
    """
    Grafica la serie de tiempo histórica del ausentismo y su proyección a 4 semanas.
    """
    fig = go.Figure()
    
    # Almacén Histórico
    df_alm = pd.DataFrame(history_almacen)
    df_alm['date'] = pd.to_datetime(df_alm['week_start_date'])
    fig.add_trace(go.Scatter(
        x=df_alm['date'],
        y=df_alm['absenteeism_rate'] * 100,
        mode='lines',
        name='Histórico Almacén (Logística)',
        line=dict(color='#E15759', width=2)
    ))
    
    # Ventas Histórico
    df_ven = pd.DataFrame(history_ventas)
    df_ven['date'] = pd.to_datetime(df_ven['week_start_date'])
    fig.add_trace(go.Scatter(
        x=df_ven['date'],
        y=df_ven['absenteeism_rate'] * 100,
        mode='lines',
        name='Histórico Ventas Técnicas',
        line=dict(color='#4E79A7', width=2)
    ))
    
    # Si hay proyecciones de Almacén, graficarlas en líneas punteadas
    if forecast_almacen:
        last_date = df_alm['date'].max()
        forecast_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(len(forecast_almacen))]
        
        # Conectar el histórico con el forecast
        f_dates = [last_date] + forecast_dates
        f_values = [df_alm['absenteeism_rate'].iloc[-1] * 100] + [f * 100 for f in forecast_almacen]
        
        fig.add_trace(go.Scatter(
            x=f_dates,
            y=f_values,
            mode='lines+markers',
            name='Pronóstico Almacén (LSTM)',
            line=dict(color='#E15759', width=2.5, dash='dash')
        ))
        
    # Si hay proyecciones de Ventas, graficarlas en líneas punteadas
    if forecast_ventas:
        last_date = df_ven['date'].max()
        forecast_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(len(forecast_ventas))]
        
        f_dates = [last_date] + forecast_dates
        f_values = [df_ven['absenteeism_rate'].iloc[-1] * 100] + [f * 100 for f in forecast_ventas]
        
        fig.add_trace(go.Scatter(
            x=f_dates,
            y=f_values,
            mode='lines+markers',
            name='Pronóstico Ventas (LSTM)',
            line=dict(color='#4E79A7', width=2.5, dash='dash')
        ))
        
    fig.update_layout(
        title=' Tendencia de Ausentismo Semanal y Pronóstico LSTM (Próximas 4 Semanas)',
        xaxis_title='Fecha',
        yaxis_title='Tasa de Ausentismo (%)',
        template='plotly_white',
        height=400,
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_shap_contributions(shap_values):
    """
    Dibuja un SHAP Waterfall plot interactivo que muestra cómo cada feature
    empuja la predicción desde el valor base del modelo hasta la predicción individual.
    Valores rojos incrementan el riesgo de fuga, valores azules lo reducen.
    """
    features = list(shap_values.keys())
    values = list(shap_values.values())

    df = pd.DataFrame({'feature': features, 'shap': values})
    df['abs_shap'] = df['shap'].abs()
    df = df.sort_values('abs_shap', ascending=False)

    # Construir el waterfall: acumulado de SHAP values
    cumulative = 0.0
    x_vals = []
    y_vals = []
    measure = []
    colors = []
    texts = []

    for _, row in df.iterrows():
        x_vals.append(row['feature'])
        y_vals.append(row['shap'])
        measure.append('relative')
        colors.append('#E15759' if row['shap'] > 0 else '#4E79A7')
        sign = '+' if row['shap'] > 0 else ''
        texts.append(f"{sign}{row['shap']:.4f}")
        cumulative += row['shap']

    fig = go.Figure(go.Waterfall(
        name='SHAP Waterfall',
        orientation='v',
        measure=measure,
        x=x_vals,
        y=y_vals,
        text=texts,
        textposition='outside',
        connector={'line': {'color': 'rgba(150,150,150,0.4)', 'width': 1}},
        increasing={'marker': {'color': '#E15759'}},
        decreasing={'marker': {'color': '#4E79A7'}},
        totals={'marker': {'color': '#2D3047'}},
    ))

    fig.update_layout(
        title='SHAP Waterfall: Impacto Individual de Factores en el Flight Risk',
        xaxis_title='Feature',
        yaxis_title='Contribución SHAP al Riesgo',
        template='plotly_white',
        height=max(380, 60 + 35 * len(features)),
        margin=dict(l=20, r=20, t=70, b=100),
        xaxis={'tickangle': -35},
    )
    return fig


def plot_shap_bar(shap_values):
    """Alternativa: gráfico de barras horizontal SHAP (para compatibilidad)."""
    features = list(shap_values.keys())
    values = list(shap_values.values())
    df = pd.DataFrame({'Característica': features, 'SHAP Value': values})
    df['Abs SHAP'] = df['SHAP Value'].abs()
    df = df.sort_values(by='Abs SHAP', ascending=True)
    df['Color'] = df['SHAP Value'].apply(lambda x: '#E15759' if x > 0 else '#4E79A7')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Característica'], x=df['SHAP Value'], orientation='h',
        marker_color=df['Color'],
        text=df['SHAP Value'].apply(lambda x: f"+{x:.4f}" if x > 0 else f"{x:.4f}"),
        textposition='outside'
    ))
    max_val = df['SHAP Value'].max()
    min_val = df['SHAP Value'].min()
    padding = max(abs(max_val), abs(min_val)) * 0.25
    fig.update_xaxes(range=[min_val - padding, max_val + padding])
    fig.update_layout(
        title='Explicabilidad SHAP: Impacto de Factores en el Flight Risk',
        xaxis_title='<- Reduce Riesgo  |  Incrementa Riesgo ->',
        yaxis_title=None, template='plotly_white',
        height=min(450, 100 + 25 * len(features)),
        margin=dict(l=20, r=20, t=60, b=40)
    )
    return fig

def plot_monte_carlo_distribution(sim_results, p5, p95, avg):
    """
    Crea un histograma de los ahorros netos de Monte Carlo mostrando percentiles críticos.
    """
    fig = go.Figure()
    
    # Histograma
    fig.add_trace(go.Histogram(
        x=sim_results,
        nbinsx=60,
        name='Distribución del Ahorro',
        marker_color='#59A14F',
        opacity=0.75
    ))
    
    # Línea promedio
    fig.add_vline(x=avg, line_width=3, line_dash="dash", line_color="black", 
                  annotation_text=f"Ahorro Promedio: ${avg:,.0f} MXN", annotation_position="top right")
                  
    # Línea P5 (Peor caso - 5to percentil)
    fig.add_vline(x=p5, line_width=2.5, line_color="#E15759", line_dash="dot",
                  annotation_text=f"Peor Escenario (P5): ${p5:,.0f} MXN", annotation_position="top left")
                  
    # Línea P95 (Mejor caso - 95vo percentil)
    fig.add_vline(x=p95, line_width=2.5, line_color="#499894", line_dash="dot",
                  annotation_text=f"Mejor Escenario (P95): ${p95:,.0f} MXN", annotation_position="top right")
                  
    fig.update_layout(
        title=' Simulación Monte Carlo: Distribución de Ahorros Netos Proyectados',
        xaxis_title='Ahorro Neto Proyectado (MXN)',
        yaxis_title='Frecuencia (Iteraciones)',
        template='plotly_white',
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )
    return fig

def plot_roi_gauge(prob_positive_roi):
    """
    Dibuja un tacómetro / gauge para la probabilidad de tener un ROI positivo.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_positive_roi,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Retorno Financiero Positivo (ROI > 0%)", 'font': {'size': 16}},
        number = {'suffix': "%", 'font': {'size': 36}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "#4E79A7"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FFC7C7'},
                {'range': [50, 80], 'color': '#FFEBA5'},
                {'range': [80, 100], 'color': '#D2EBD4'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white'
    )
    return fig


# ============================================================
# NUEVOS GRÁFICOS DE ANÁLISIS AVANZADO v2.0
# ============================================================

def plot_cohort_heatmap(cohorts_data):
    """
    Heatmap de riesgo de rotación por cohorte de antigüedad vs departamento.
    """
    df = pd.DataFrame(cohorts_data)
    if df.empty:
        return go.Figure()

    pivot = df.pivot_table(
        values='turnover_rate_pct', index='role', columns='cohort', aggfunc='first'
    )
    cohort_order = ['0-6 meses', '6-12 meses', '1-2 años', '2-5 años', '5+ años']
    pivot = pivot.reindex(columns=[c for c in cohort_order if c in pivot.columns])

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, '#D2EBD4'],
            [0.4, '#FFEBA5'],
            [0.7, '#FFB347'],
            [1.0, '#E15759'],
        ],
        text=[[f"{v:.1f}%" if not np.isnan(v) else "N/A" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 13},
        colorbar=dict(title="Tasa de<br>Rotación (%)"),
        hoverongaps=False,
    ))

    fig.update_layout(
        title='Heatmap de Rotación por Cohorte de Antigüedad y Departamento',
        xaxis_title='Cohorte de Antigüedad',
        yaxis_title='Departamento',
        template='plotly_white',
        height=280,
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def plot_cohort_bars(cohorts_data):
    """
    Barras agrupadas: Bradford Factor promedio y Carga de Trabajo por cohorte.
    """
    df = pd.DataFrame(cohorts_data)
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    for role, grp in df.groupby('role'):
        color = '#E15759' if role == 'Almacén' else '#4E79A7'
        fig.add_trace(go.Bar(
            x=grp['cohort'], y=grp['avg_bradford_factor'],
            name=f'Bradford Factor — {role}',
            marker_color=color, opacity=0.85,
        ))

    fig.update_layout(
        title='Bradford Factor Promedio por Cohorte de Antigüedad',
        xaxis_title='Cohorte de Antigüedad',
        yaxis_title='Bradford Factor Promedio',
        barmode='group',
        template='plotly_white',
        height=320,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_survival_curves(survival_data):
    """
    Grafica las curvas de supervivencia Kaplan-Meier por grupo.
    """
    fig = go.Figure()
    color_map = {
        'Global': '#2D3047',
        'Almacén': '#E15759',
        'Ventas Técnicas': '#4E79A7',
    }
    dash_map = {'Global': 'solid', 'Almácn': 'dash', 'Ventas Técnicas': 'dot'}

    for curve in survival_data.get('curves', []):
        group = curve['group']
        points = pd.DataFrame(curve['points'])
        color = color_map.get(group, '#59A14F')
        dash = dash_map.get(group, 'solid')

        # Banda de confianza 95%
        fig.add_trace(go.Scatter(
            x=list(points['month']) + list(points['month'][::-1]),
            y=list(points['ci_upper']) + list(points['ci_lower'][::-1]),
            fill='toself',
            fillcolor=color.replace(')', ', 0.1)').replace('rgb', 'rgba') if 'rgb' in color else color + '22',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
            name=f'CI 95% {group}',
        ))

        # Curva principal
        fig.add_trace(go.Scatter(
            x=points['month'],
            y=points['survival_prob'],
            mode='lines',
            name=group,
            line=dict(color=color, width=2.5, dash=dash),
            hovertemplate=f'<b>{group}</b><br>Mes %{{x}}<br>P(Supervivencia): %{{y:.2%}}<extra></extra>',
        ))

    # Línea horizontal en 50% (mediana)
    fig.add_hline(
        y=0.5, line_dash='dash', line_color='gray', opacity=0.6,
        annotation_text='Mediana de supervivencia (50%)', annotation_position='bottom right'
    )

    fig.update_layout(
        title='Análisis de Supervivencia Kaplan-Meier — Tiempo Hasta la Renuncia',
        xaxis_title='Antigüedad (Meses)',
        yaxis_title='Probabilidad de Supervivencia (Retención)',
        yaxis=dict(range=[0, 1.05], tickformat='.0%'),
        template='plotly_white',
        height=420,
        margin=dict(l=40, r=40, t=70, b=40),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def plot_anomaly_scatter(anomalies_data):
    """
    Scatter plot de anomalías: Bradford Factor vs Horas Extra,
    coloreado por nivel de anomalía.
    """
    df = pd.DataFrame(anomalies_data)
    if df.empty:
        return go.Figure()

    color_map = {
        'NORMAL': '#59A14F',
        'SOSPECHOSO': '#F28E2B',
        'ANOMALÍA CRÍTICA': '#E15759',
    }
    symbol_map = {
        'NORMAL': 'circle',
        'SOSPECHOSO': 'diamond',
        'ANOMALÍA CRÍTICA': 'x',
    }

    fig = go.Figure()
    for label, grp in df.groupby('anomaly_label'):
        fig.add_trace(go.Scatter(
            x=grp['overtime_hours_last_month'],
            y=grp['bradford_factor'],
            mode='markers',
            name=label,
            marker=dict(
                color=color_map.get(label, '#999'),
                symbol=symbol_map.get(label, 'circle'),
                size=grp['anomaly_percentile'].clip(6, 20),
                opacity=0.8,
                line=dict(width=0.5, color='white'),
            ),
            customdata=grp[['name', 'role', 'anomaly_percentile', 'workload_index']].values,
            hovertemplate=(
                '<b>%{customdata[0]}</b> (%{customdata[1]})<br>'
                'Horas Extra: %{x:.1f} hrs<br>'
                'Bradford Factor: %{y:.0f}<br>'
                'Percentil Anomalía: %{customdata[2]:.1f}<br>'
                'Carga Trabajo: %{customdata[3]:.1f}<extra></extra>'
            ),
        ))

    fig.update_layout(
        title='Detección de Anomalías: Bradford Factor vs Horas Extra (Isolation Forest)',
        xaxis_title='Horas Extra (Mes)',
        yaxis_title='Índice de Bradford (Severidad de Ausentismo)',
        template='plotly_white',
        height=420,
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(title='Clasificación'),
    )
    return fig


def plot_retention_radar(action_probabilities, recommended_action):
    """
    Radar chart de las probabilidades de cada intervención de retención.
    """
    labels_map = {
        'aumento_salarial': 'Aumento Salarial',
        'reducir_horas_extra': 'Reducir Horas Extra',
        'capacitacion': 'Capacitación',
        'mentoria_liderazgo': 'Mentoría / Liderazgo',
        'reubicacion_departamento': 'Reubicación',
    }
    actions = list(action_probabilities.keys())
    probs = [action_probabilities[a] * 100 for a in actions]
    labels = [labels_map.get(a, a) for a in actions]

    fig = go.Figure(go.Scatterpolar(
        r=probs + [probs[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(78, 121, 167, 0.25)',
        line=dict(color='#4E79A7', width=2),
        name='Probabilidad de Intervención',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickformat='.0f', ticksuffix='%'),
        ),
        title=f'Intervención Recomendada: <b>{labels_map.get(recommended_action, recommended_action)}</b>',
        showlegend=False,
        template='plotly_white',
        height=320,
        margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig
