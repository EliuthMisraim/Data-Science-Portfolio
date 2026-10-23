import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuración del Path y Entorno
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pca_engine import PCAProcessor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------
# Configuración de la Página de Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="PCA Explorer | Módulo 28",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una interfaz pulida y moderna
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 5px solid #4361ee;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        color: #1e293b;
        font-weight: 700;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 18px;
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        border: 1px solid #c7d2fe;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Funciones de Carga y Entrenamiento en Caché
# ---------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    """Carga el dataset de Iris desde data/raw y normaliza nombres de columnas."""
    data_path = ROOT_DIR / "data" / "raw" / "iris.csv"
    if not data_path.exists():
        # Enlace raw de respaldo si el archivo local no estuviera presente
        url = "https://gist.githubusercontent.com/netj/8836201/raw/iris.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(data_path)
    
    df.columns = df.columns.str.replace('.', '_', regex=False)
    return df

@st.cache_resource
def train_pca_model(df, feature_cols):
    """Entrena y ajusta el motor modular PCAProcessor con 2 componentes."""
    processor = PCAProcessor(n_components=2)
    projections = processor.fit_transform(df, feature_cols)
    return processor, projections

# Carga de datos y modelo
df_iris = load_and_prepare_data()
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

pca_processor, X_pca = train_pca_model(df_iris, numeric_cols)
df_iris['PC1'] = X_pca[:, 0]
df_iris['PC2'] = X_pca[:, 1]
var_exp = pca_processor.explained_variance_ratio_
loadings = pca_processor.get_loadings()

# Centroides en el espacio PCA por variedad (usan las etiquetas 'variety')
cluster_centroids = df_iris.groupby('variety')[['PC1', 'PC2']].mean()

# ---------------------------------------------------------
# Barra Lateral (Sidebar): Parámetros Biométricos
# ---------------------------------------------------------
st.sidebar.header("🌸 Parámetros de la Flor")
st.sidebar.markdown(
    "Ingresa las 4 mediciones biométricas de una nueva observación para proyectarla en tiempo real "
    "sobre el mapa de componentes principales."
)

# Presets de flores conocidas
preset_choice = st.sidebar.selectbox(
    "Plantillas de muestra rápida:",
    [
        "Personalizado (Ajustar deslizadores)",
        "Típica Setosa (Pétalos pequeños)",
        "Típica Versicolor (Medidas intermedias)",
        "Típica Virginica (Pétalos grandes)"
    ]
)

presets = {
    "Típica Setosa (Pétalos pequeños)": (5.0, 3.4, 1.5, 0.2),
    "Típica Versicolor (Medidas intermedias)": (5.9, 2.7, 4.2, 1.3),
    "Típica Virginica (Pétalos grandes)": (6.5, 3.0, 5.5, 2.0)
}

if preset_choice in presets:
    init_sl, init_sw, init_pl, init_pw = presets[preset_choice]
else:
    init_sl, init_sw, init_pl, init_pw = 5.8, 3.0, 4.3, 1.3

sl = st.sidebar.slider("Longitud del Sépalo (cm)", 4.0, 8.0, float(init_sl), 0.1)
sw = st.sidebar.slider("Ancho del Sépalo (cm)", 2.0, 4.5, float(init_sw), 0.1)
pl = st.sidebar.slider("Longitud del Pétalo (cm)", 1.0, 7.0, float(init_pl), 0.1)
pw = st.sidebar.slider("Ancho del Pétalo (cm)", 0.1, 2.5, float(init_pw), 0.1)

# Construir DataFrame de la nueva muestra y proyectarla
sample_df = pd.DataFrame([{
    'sepal_length': sl,
    'sepal_width': sw,
    'petal_length': pl,
    'petal_width': pw
}])

sample_pca = pca_processor.transform(sample_df, numeric_cols)
sample_pc1 = sample_pca[0, 0]
sample_pc2 = sample_pca[0, 1]

# Cálculo de distancias Euclidianas a los centroides
distances = {}
for variety, centroid in cluster_centroids.iterrows():
    dist = np.sqrt((sample_pc1 - centroid['PC1'])**2 + (sample_pc2 - centroid['PC2'])**2)
    distances[variety] = dist

predicted_variety = min(distances, key=distances.get)

# ---------------------------------------------------------
# Encabezado Principal y KPIs
# ---------------------------------------------------------
st.title("🚀 PCA Explorer: Reducción de Dimensiones en Producción")
st.markdown(
    "Esta aplicación interactiva despliega en tiempo real el pipeline modular de estandarización "
    "y **Análisis de Componentes Principales (PCA)** desarrollado en `src/pca_engine.py`."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Varianza PC1</div>
        <div class="metric-value">{var_exp[0]*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Varianza PC2</div>
        <div class="metric-value">{var_exp[1]*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Varianza Total Explicada</div>
        <div class="metric-value">{(var_exp[0] + var_exp[1])*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Proyección de Muestra</div>
        <div class="metric-value">[{sample_pc1:.2f}, {sample_pc2:.2f}]</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Cuadro de diagnóstico no supervisado
st.markdown(f"""
<div class="prediction-box">
    <h4 style="margin: 0 0 8px 0; color: #3730a3;">🧭 Clasificación por Proximidad a Centroides</h4>
    <p style="margin: 0; font-size: 1.05rem; color: #1e1b4b;">
        La nueva muestra se sitúa más próxima al centroide de <strong>Iris {predicted_variety}</strong> 
        (Distancia euclidiana en plano PCA: <code>{distances[predicted_variety]:.2f}</code>).<br>
        <span style="font-size: 0.85rem; color: #4338ca;">PCA es no supervisado, pero los centroides se calculan
        con las etiquetas de variedad, por lo que esta regla es un clasificador sencillo. En 2D,
        Versicolor y Virginica se solapan: úsala como orientación, no como diagnóstico.</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Pestañas de Análisis Visual
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📍 Mapa de Observaciones (Proyección en Vivo)", 
    "🧭 Mapa de Factores (Cargas Factoriales)", 
    "📋 Datos y Explicabilidad"
])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Paleta de colores para las especies
    colors = {'Setosa': '#3b82f6', 'Versicolor': '#10b981', 'Virginica': '#8b5cf6'}
    
    # 1. Puntos del dataset histórico
    for variety_name, group in df_iris.groupby('variety'):
        ax.scatter(
            group['PC1'], group['PC2'], 
            label=f'Iris {variety_name}',
            color=colors.get(variety_name, '#64748b'),
            alpha=0.65, s=60, edgecolors='none'
        )
        
    # 2. Centroides de cada grupo
    for variety_name, centroid in cluster_centroids.iterrows():
        ax.scatter(
            centroid['PC1'], centroid['PC2'],
            marker='X', s=160, color='black', alpha=0.8,
            label=f'Centroide {variety_name}' if variety_name == 'Setosa' else None
        )
        ax.text(
            centroid['PC1'] + 0.1, centroid['PC2'] + 0.1, 
            f"μ_{variety_name}", fontsize=11, fontweight='bold', color='#1e293b'
        )

    # 3. Muestra ingresada por el usuario
    ax.scatter(
        sample_pc1, sample_pc2, 
        color='#ef4444', s=250, marker='*', edgecolors='#7f1d1d', linewidths=1.5,
        label='Tu Muestra Ingresada', zorder=10
    )
    ax.annotate(
        f"Tu Flor\n({sample_pc1:.2f}, {sample_pc2:.2f})",
        xy=(sample_pc1, sample_pc2),
        xytext=(sample_pc1 + 0.4, sample_pc2 + 0.4),
        arrowprops=dict(facecolor='#ef4444', shrink=0.08, width=1.5, headwidth=6),
        fontsize=11, fontweight='bold', color='#b91c1c'
    )
    
    ax.set_title("Mapa de Observaciones con Proyección en Tiempo Real", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Componente Principal 1 ({var_exp[0]*100:.2f}% de varianza)", fontsize=11)
    ax.set_ylabel(f"Componente Principal 2 ({var_exp[1]*100:.2f}% de varianza)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)
    
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig_f, ax_f = plt.subplots(figsize=(8, 6))
    
    ax_f.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax_f.axvline(0, color='grey', linestyle='--', linewidth=1)
    
    # Círculo de correlación unitario (válido porque se grafican cargas = correlaciones)
    circle = plt.Circle((0, 0), 1.0, color='lightblue', fill=False, linestyle=':', linewidth=1.5)
    ax_f.add_artist(circle)
    
    label_offsets = {'petal_width': (0, 0.07), 'petal_length': (0, -0.07)}
    for feature in loadings.columns:
        x_val = loadings.loc['PC1', feature]
        y_val = loadings.loc['PC2', feature]
        ax_f.arrow(0, 0, x_val, y_val, color='#dc2626', alpha=0.85, head_width=0.04, head_length=0.06,
                   length_includes_head=True)
        dx, dy = label_offsets.get(feature, (0, 0))  # separa etiquetas que se encimarían
        ax_f.text(x_val * 1.1 + dx, y_val * 1.1 + dy, feature, color='#1d4ed8', fontsize=11, fontweight='bold', ha='center')
        
    ax_f.set_xlim(-1.25, 1.25)
    ax_f.set_ylim(-1.25, 1.25)
    ax_f.set_aspect('equal')
    ax_f.set_title("Círculo de Correlación (Cargas de las Variables Originales)", fontsize=14, fontweight='bold')
    ax_f.set_xlabel(f"Carga en PC1 ({var_exp[0]*100:.2f}%)", fontsize=11)
    ax_f.set_ylabel(f"Carga en PC2 ({var_exp[1]*100:.2f}%)", fontsize=11)
    ax_f.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig_f)
    plt.close(fig_f)
    
    st.markdown("""
    **Interpretación de los Vectores (cargas = correlación variable-componente):**
    - Variables que apuntan en la misma dirección están correlacionadas positivamente (`petal_length` y `petal_width`).
    - Cuanto más cerca del círculo (longitud ≈ 1), mejor representada está la variable en este plano.
    - `petal_length`, `petal_width` y `sepal_length` cargan sobre PC1; `sepal_width` domina PC2.
    """)

with tab3:
    st.subheader("Cargas Factoriales (correlación variable-componente)")
    st.dataframe(loadings.style.format("{:.4f}").background_gradient(cmap='coolwarm', axis=None), width="stretch")
    
    st.subheader("Eigenvectores (dirección de cada componente)")
    st.dataframe(pca_processor.get_factor_map().style.format("{:.4f}").background_gradient(cmap='coolwarm', axis=None), width="stretch")
    
    st.subheader("Muestra Actual vs. Estadísticos de Población")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("**Valores de Entrada:**")
        st.dataframe(sample_df, width="stretch")
    with col_s2:
        st.write("**Distancias a Centroides en Plano PCA:**")
        dist_df = pd.DataFrame([distances], index=["Distancia Euclidiana"])
        st.dataframe(dist_df.style.format("{:.3f}"), width="stretch")
