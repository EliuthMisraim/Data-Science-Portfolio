# 📘 Guía Maestra: Arquitectura de Proyectos de Data Science, PCA y Storytelling de Negocio

Esta guía está diseñada como un recurso de estudio y referencia profesional para entender cómo se construye un proyecto de ciencia de datos robusto de extremo a extremo (*End-to-End*), cómo dominar el algoritmo **PCA (Principal Component Analysis)** y cómo traducir conceptos matemáticos avanzados en **valor de negocio cuantificable**.

---

## 🏗️ 1. Arquitectura de un Proyecto Profesional de Data Science

Uno de los saltos más importantes entre un principiante y un **Científico de Datos Senior / ML Engineer** es abandonar el *"anti-patrón del notebook monolítico"* (escribir miles de líneas en un solo archivo `.ipynb`).

En la industria se utiliza una **arquitectura desacoplada en 4 capas**:

```text
proyecto_ml/
│
├── data/                    # CAPA DE ALMACENAMIENTO (Inmutable)
│   ├── raw/                 # Datos brutos originales (ej. iris.csv)
│   └── processed/           # Datos limpios y transformados (opcional; no se usa en este módulo)
│
├── src/                     # CAPA DE INGENIERÍA (Lógica pura y reutilizable)
│   ├── __init__.py          # Convierte la carpeta en paquete de Python
│   └── pca_engine.py        # Clases modulares (ej. PCAProcessor con fit/transform)
│
├── notebooks/               # CAPA DE STORYTELLING Y VALIDACIÓN (Exploración)
│   └── 01_estudio_pca.ipynb # Narrativa analítica e hipótesis (importa desde src/)
│
├── app/                     # CAPA DE PRESENTACIÓN / DESPLIEGUE (Consumo de valor)
│   └── app.py               # Aplicación interactiva (Streamlit) para stakeholders
│
├── requirements.txt         # Dependencias exactas del proyecto
└── README.md                # Documentación ejecutiva del proyecto
```

### 1.1 Responsabilidad de Cada Directorio

| Carpeta | Rol Principal | Regla de Oro |
| :--- | :--- | :--- |
| `data/raw/` | Almacenamiento de datos brutos. | **Inmutabilidad absoluta.** Ningún script debe modificar ni sobreescribir los archivos de esta carpeta. |
| `data/processed/` | Datos limpios o transformados. | Son reproducibles; si se borran, un script los puede volver a generar. |
| `src/` | **Capa de lógica e ingeniería de software.** Clases y funciones reutilizables. | No debe contener código de visualización ad-hoc ni celdas exploratorias. Código testeable con tipado estático (`typing`). |
| `notebooks/` | **Storytelling, exploración y validación.** | No debe definir funciones o algoritmos pesados; debe consumirlos de `src/` para mantener las celdas limpias y enfocadas en la interpretación. |
| `app/` | **Demostración y consumo de valor.** | Permite a perfiles no técnicos interactuar con el modelo en tiempo real. |

> **El Principio DRY (*Don't Repeat Yourself*):**
> Al encapsular la lógica matemática en `src/pca_engine.py`, tanto el notebook como la app de Streamlit consumen exactamente la misma clase `PCAProcessor`. Si mañana cambias la lógica o agregas una nueva métrica, ambos entornos se actualizan automáticamente sin duplicar código.

---

## 🔬 2. Reducción de Dimensiones: Análisis de Componentes Principales (PCA)

### 2.1 El Problema: ¿Por qué necesitamos reducir dimensiones?

En la analítica de datos moderna nos enfrentamos a dos problemas críticos:
1. **La Maldición de la Dimensionalidad (*Curse of Dimensionality*):** A mayor número de variables, el volumen del espacio crece exponencialmente. Los datos se vuelven "dispersos" (*sparse*), lo que incrementa el riesgo de sobreajuste (*overfitting*) y dispara los costos de cómputo.
2. **Multicolinealidad:** Muchas variables miden esencialmente el mismo fenómeno (por ejemplo: `petal_length` y `petal_width`). Mantener variables redundantes confunde a los modelos supervisados (como regresiones lineales o logísticas) al desestabilizar sus coeficientes.

---

### 2.2 ¿Qué hace PCA intuitivamente?

Imagina que tomas una fotografía 2D de una escultura tridimensional. 
- Si tomas la foto desde un mal ángulo, la escultura parecerá una masa plana e incomprensible.
- Si buscas el ángulo óptimo (donde la escultura proyecta la mayor sombra y muestra todos sus detalles), capturas casi toda la información tridimensional en una superficie plana de 2 dimensiones.

**Eso hace PCA:** Encuentra el ángulo de visión matemático donde los datos están más dispersos (tienen **mayor varianza**) para proyectarlos perdiendo la menor cantidad de información posible.

---

### 2.3 El Proceso Matemático Paso a Paso

#### Paso 1: Estandarización Obligatoria ($Z$-Score)
$$z = \frac{x - \mu}{\sigma}$$

> **¿Por qué es obligatoria la estandarización?**
> PCA busca maximizar la varianza. Si tienes una variable en metros (ej. 1.80 m) y otra en ingresos anuales (ej. $50,000 USD), la variable de ingresos tendrá numéricamente una varianza millones de veces mayor solo por la escala de medición. PCA ignoraría la altura creyendo erróneamente que los ingresos contienen "toda la información". `StandardScaler` pone a todas las variables a competir en igualdad de condiciones (media 0, desviación estándar 1).

#### Paso 2: Matriz de Covarianza / Correlaciones
Mide cómo varía cada par de variables simultáneamente. Si dos variables tienen correlación positiva alta, se mueven juntas.

#### Paso 3: Eigenvalores ($\lambda$) y Eigenvectores ($v$)
Mediante álgebra lineal (descomposición espectral):
$$\Sigma v = \lambda v$$
- **Eigenvector ($v$):** Indica la **dirección geométrica** del nuevo eje coordenado (el Componente Principal). Todos los eigenvectores son perpendiculares entre sí (ortogonales = 0% de correlación entre componentes).
- **Eigenvalor ($\lambda$):** Es un número escalar que representa la **magnitud de varianza** que absorbe ese eje.

#### Paso 4: Razón de Varianza Explicada (*Explained Variance Ratio*)
Indica qué porcentaje de la información total captura cada componente:
$$\text{Varianza Explicada (PC}_i\text{)} = \frac{\lambda_i}{\sum_{k=1}^p \lambda_k}$$

En el dataset de Iris:
- **PC1:** Explica el **72.96%** de la varianza.
- **PC2:** Explica el **22.85%** de la varianza.
- **Total acumulado (2 dimensiones):** **95.81%**. Pasamos de 4 variables a 2 conservando casi el 96% de la información del sistema original.

---

### 2.4 Interpretación Dual: Observaciones vs. Factores

1. **Mapa de Observaciones (Plano PC1 vs PC2):** Muestra cada registro (flor, cliente, transacción) como un punto. Los grupos naturales (*clusters*) tienden a aparecer separados sin necesidad de etiquetas previas (en Iris, *Setosa* se aísla claramente; *Versicolor* y *Virginica* se solapan parcialmente).
2. **Mapa de Factores (Cargas Factoriales / *Biplot*):** Muestra las variables originales representadas por flechas vectoriales:
   - **Longitud de la flecha:** Cuánta influencia tiene esa variable en el plano (cercana a 1 indica máxima representatividad). Esto solo es válido si se grafican las **cargas** (eigenvector × √eigenvalor = correlación variable-componente), no los eigenvectores crudos; por eso el proyecto expone `get_loadings()`.
   - **Ángulo entre flechas:** Ángulo agudo ($\approx 0^\circ$) indica correlación positiva fuerte; ángulo recto ($90^\circ$) indica independencia; ángulo opuesto ($180^\circ$) indica correlación negativa.

---

## 💼 3. Cómo Explicar PCA a Negocio (Traducción Ejecutiva)

A los directores generales (*CEOs*), gerentes de operaciones o clientes no les interesan los eigenvectores ni la matriz de covarianza. Les interesa el **ahorro de costos, reducción de riesgos y toma de decisiones rápida**.

### 3.1 Diccionario de Traducción: Técnico $\rightarrow$ Negocio

| Término Técnico | Lo que dices en la junta con Ejecutivos |
| :--- | :--- |
| *"Calculamos la descomposición en eigenvectores de la matriz de covarianza."* | *"Sintetizamos decenas de métricas redundantes en dos macro-indicadores estratégicos."* |
| *"La varianza explicada acumulada de PC1 y PC2 es 95.81%."* | *"Al reducir la complejidad a dos indicadores simples, retenemos el **96% de la información histórica real**, descartando solo un 4% de información de menor peso (no necesariamente ruido)."* |
| *"Las variables se proyectan en un subespacio ortogonal no supervisado."* | *"Eliminamos la duplicidad de información para que ninguna variable distorsione las decisiones con datos repetidos."* |
| *"Ubicamos la observación en el mapa factorial."* | *"Generamos un tablero visual donde podemos ubicar a cualquier cliente o producto en tiempo real para saber a qué perfil de comportamiento pertenece."* |

---

### 3.2 Casos de Uso Reales de Negocio

1. **Banca / Finanzas (Credit Scoring):**
   Un banco tiene 80 variables financieras de un solicitante de crédito (ingreso, saldo promedio, deuda en tarjeta, etc.). Muchas miden lo mismo (liquidez). PCA las resume en:
   - **PC1 (Solvencia y Liquidez General).**
   - **PC2 (Propensión al Endeudamiento Inmediato).**
2. **E-commerce / Retail (Segmentación de Clientes):**
   Un catálogo con miles de productos o perfiles de clientes puede proyectarse en 2 ejes visuales interactivos para entender qué clientes compran por conveniencia vs. quiénes compran por precio.
3. **Mantenimiento Predictivo (Detección de Anomalías e IoT):**
   Monitoreo de 20 sensores de temperatura y vibración en motores. En vez de poner 20 alarmas, se proyectan las 20 lecturas en 2 componentes principales. Si el punto cae fuera de la nube normal de puntos, se activa una alarma inmediata por comportamiento anómalo.
