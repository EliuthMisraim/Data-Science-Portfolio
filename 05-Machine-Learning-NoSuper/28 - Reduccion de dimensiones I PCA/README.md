# 📊 Módulo 28: Reducción de Dimensiones I - PCA (Principal Component Analysis)

Bienvenido al proyecto de **Análisis de Componentes Principales (PCA)**, enfocado en técnicas no supervisadas de reducción de dimensionalidad aplicadas a ciencia de datos y aprendizaje automático bajo una arquitectura modular de producción (*End-to-End*).

---

## 🎯 Objetivo y Valor de Negocio

En la analítica avanzada y el aprendizaje automático, los conjuntos de datos frecuentemente sufren de la **maldición de la dimensionalidad** (*Curse of Dimensionality*) y de **multicolinealidad** entre variables. Esto genera:
- Costos computacionales elevados e ineficiencia en el entrenamiento de modelos.
- Dificultad para visualizar patrones y relaciones complejas en más de 3 dimensiones.
- Riesgo de sobreajuste (*overfitting*) al incorporar ruido innecesario.

### Impacto y Retorno de Inversión (ROI):
1. **Eficiencia Operativa**: En Iris se pasa de 4 a 2 variables reteniendo el **95.8% de la varianza** (complejidad a la mitad). Con 4 variables el ahorro de cómputo es simbólico; el beneficio real aparece en datasets con decenas o cientos de variables.
2. **Interpretabilidad y Visualización**: Proyección de perfiles de clientes, transacciones o variables de negocio en planos 2D comprensibles para la toma de decisiones ejecutiva.
3. **Calidad y Robustez del Modelo**: Mitigación de redundancias y colinealidad previa a modelos supervisados (regresiones, clasificadores) o de segmentación (clustering).

---

## 📈 Resultados Clave del Modelo (Dataset Iris)

- **Variables Originales Analizadas:** 4 (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).
- **Componentes Retenidos:** 2 componentes principales ($k = 2$).
- **Varianza Explicada por PC1:** **72.96%** (Captura el tamaño general de pétalo y longitud de sépalo; cargas: `petal_length` 0.99, `petal_width` 0.97, `sepal_length` 0.89).
- **Varianza Explicada por PC2:** **22.85%** (Captura predominantemente el ancho de sépalo; carga de `sepal_width` 0.89).
- **Varianza Total Acumulada:** **95.81%** (Pérdida de señal menor al 4.2%).
- **Diagnóstico:** El PCA (no supervisado) separa a *Setosa* de forma natural en el eje PC1 por sus reducidas dimensiones de pétalo. *Versicolor* y *Virginica* se solapan parcialmente: una regla de centroide más cercano en el plano 2D acierta 85.3% (evaluado sobre los mismos datos con que se ajustó, y usando las etiquetas para calcular los centroides).

---

## 📁 Estructura del Entorno de Trabajo

El proyecto adopta un estándar de ingeniería desacoplado en 4 capas para evitar el anti-patrón del notebook monolítico:

```text
28 - Reduccion de dimensiones I PCA/
│
├── data/
│   └── raw/
│       └── iris.csv         # Datos originales inmutables (150 registros)
│
├── src/                     # ⚙️ CAPA DE INGENIERÍA Y DOMINIO
│   ├── __init__.py          # Inicializador de paquete Python
│   └── pca_engine.py        # Clase modular PCAProcessor (fit_transform, transform, cargas)
│
├── notebooks/               # 📓 CAPA DE STORYTELLING Y EXPLORACIÓN
│   └── 01_estudio_pca.ipynb # Cuaderno de análisis, correlaciones y diagnósticos
│
├── app/                     # 🌐 CAPA DE DESPLIEGUE INTERACTIVO
│   └── app.py               # Aplicación web en Streamlit para inferencia en tiempo real
│
├── docs/                    # 📚 DOCUMENTACIÓN Y GUÍAS DE ESTUDIO
│   └── guia_estudio_pca_arquitectura.md # Guía completa de arquitectura, teoría y negocio
│
├── requirements.txt         # Dependencias reproducibles del entorno
├── .gitignore               # Excluye __pycache__, entornos virtuales, checkpoints
└── README.md                # Portada ejecutiva del proyecto
```

---

## 🚀 Instalación y Uso

### 1. Crear y activar entorno virtual (Recomendado)
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecución por Capas

#### A. Storytelling Analítico (Jupyter Notebook)
Abre VS Code o Jupyter Lab y ejecuta el cuaderno:
```bash
# O abrirlo directamente desde tu IDE
notebooks/01_estudio_pca.ipynb
```

#### B. Capa de Ingeniería Modular (Scripts / Python)
Reutiliza la lógica de preprocesamiento y PCA directamente en tus pipelines:
```python
from src.pca_engine import PCAProcessor

# Inicializar motor
pca = PCAProcessor(n_components=2)

# Ajuste y transformación histórica
X_pca = pca.fit_transform(df, feature_columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Inferencia de nuevas observaciones en producción (sin refit)
nuevos_puntos_pca = pca.transform(df_nuevos, feature_columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Cargas factoriales (correlación variable-componente) para el círculo de correlación
cargas = pca.get_loadings()
```

#### C. Aplicación Web Interactiva (Streamlit)
Lanza el simulador visual para interactuar con deslizadores y clasificar muestras en tiempo real:
```bash
streamlit run app/app.py
```

---

## 📚 Documentación Adicional
Para profundizar en los fundamentos matemáticos (eigenvalores, eigenvectores, matriz de covarianza) y en cómo vender este tipo de proyectos ante comités directivos, consulta la [Guía Maestra de Estudio](docs/guia_estudio_pca_arquitectura.md).
