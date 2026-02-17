# Data Science Portfolio

Este repositorio contiene mi portafolio de Ciencia de Datos, organizado por temas y tecnologías.

## Estructura de Carpetas

- **01-Python-Fundamentos**: Conceptos básicos de Python.
- **02-Librerias-Analisis**: Pandas, NumPy, Matplotlib, etc.
- **03-Metodologias-Data**: Procesos y metodologías de Ciencia de Datos.
- **04-Machine-Learning-Super**: Modelos supervisados (Regresión, Clasificación).
- **05-Machine-Learning-NoSuper**: Modelos no supervisados (Clustering, PCA).
- **06-Estadistica**: Conceptos y aplicaciones estadísticas.
- **07-SQL-Data-Analysis**: Scripts y consultas SQL.
- **08-Power-BI**: Dashboards y reportes.
- **09-Series-de-Tiempo**: Análisis temporal y forecasting.

## Cómo Actualizar el Portafolio

He incluido un script de automatización para facilitar la subida de nuevos archivos a GitHub.

### Opción 1: Ejecución Rápida (Recomendada)
Haz doble clic en el archivo **`run_update.bat`**.  
Esto abrirá una ventana, sincronizará tus archivos y se cerrará automáticamente al terminar.

### Opción 2: PowerShell
1. Haz clic derecho en el archivo `update_portfolio.ps1`.
2. Selecciona **"Ejecutar con PowerShell"**.

### Opción 3: Línea de Comandos
Abre una terminal en esta carpeta y ejecuta:
```powershell
.\update_portfolio.ps1
```
Puedes agregar un mensaje de commit personalizado:
```powershell
.\update_portfolio.ps1 "Agregué nuevo análisis de ventas"
```
