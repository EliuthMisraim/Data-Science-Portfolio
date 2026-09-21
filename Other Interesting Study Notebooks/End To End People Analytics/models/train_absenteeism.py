import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Establecer semilla para reproducibilidad
np.random.seed(42)

def train_model():
    db_path = 'data/people_analytics.db'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(db_path):
        print(f"Error: No se encontró la base de datos en {db_path}.")
        return

    print("Conectando a la base de datos para extraer histórico de ausentismo...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM absenteeism_history ORDER BY week_start_date ASC", conn)
    conn.close()
    
    print(f"Historial de ausentismo cargado: {len(df)} registros.")
    
    # 1. Ingeniería de Características de Series Temporales
    # Agregar variables de fecha y estacionalidad cíclica
    df['date_dt'] = pd.to_datetime(df['week_start_date'])
    df['week_of_year'] = df['date_dt'].dt.isocalendar().week
    
    # Transformación Seno/Coseno para estacionalidad
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52.0)
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52.0)
    
    # Codificar departamento: Almacén -> 0, Ventas Técnicas -> 1
    df['dept_encoded'] = df['department'].map({'Almacén': 0, 'Ventas Técnicas': 1})
    
    # 2. Escalamiento de la Variable de Tasa de Ausentismo (MinMax)
    rate_min = df['absenteeism_rate'].min()
    rate_max = df['absenteeism_rate'].max()
    
    if rate_max == rate_min:
        rate_max += 1e-5
        
    scaler = {
        'rate_min': float(rate_min),
        'rate_max': float(rate_max)
    }
    
    with open(os.path.join(models_dir, 'absenteeism_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    df['rate_scaled'] = (df['absenteeism_rate'] - rate_min) / (rate_max - rate_min)
    
    # 3. Preparación de Ventanas Deslizantes (Sliding Windows)
    # Ventana de entrada (lookback) = 8 semanas
    # Horizonte de predicción = 4 semanas
    lookback = 8
    horizon = 4
    
    X_list = []
    y_list = []
    
    # Hacer el proceso por departamento para no mezclar las series
    for dept in ['Almacén', 'Ventas Técnicas']:
        df_dept = df[df['department'] == dept].copy().reset_index(drop=True)
        n_records = len(df_dept)
        
        for i in range(n_records - lookback - horizon + 1):
            # Historial de tasas (lookback)
            rates = df_dept.loc[i : i + lookback - 1, 'rate_scaled'].values
            # Historial de componentes de estacionalidad
            sins = df_dept.loc[i : i + lookback - 1, 'sin_week'].values
            coss = df_dept.loc[i : i + lookback - 1, 'cos_week'].values
            
            # Departamento constante para toda la ventana
            dept_code = df_dept.loc[i, 'dept_encoded']
            
            # Aplanar vector de características: tasas + senos + cosenos + dept
            feature_vector = np.concatenate([rates, sins, coss, [dept_code]])
            
            # Target de salida: las siguientes 4 semanas de tasa escalada
            target_vector = df_dept.loc[i + lookback : i + lookback + horizon - 1, 'rate_scaled'].values
            
            X_list.append(feature_vector)
            y_list.append(target_vector)
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Estructura de Datos para Red Neuronal - X: {X.shape}, y: {y.shape}")
    
    # 4. Partición Entrenamiento y Validación (85/15)
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 5. Instanciar y Entrenar Red Neuronal (MLPRegressor)
    print("Entrenando Red Neuronal MLPRegressor (Deep Learning / Feedforward)...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.005,
        max_iter=600,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    model.fit(X_train, y_train)
    
    # 6. Evaluación
    y_pred = model.predict(X_val)
    
    # Des-escalar para evaluar métricas en escala real (%)
    y_val_real = y_val * (rate_max - rate_min) + rate_min
    y_pred_real = y_pred * (rate_max - rate_min) + rate_min
    
    mse = mean_squared_error(y_val_real, y_pred_real)
    mae = mean_absolute_error(y_val_real, y_pred_real)
    
    print("\n--- Resultados de la Red Neuronal en Conjunto de Validación ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae*100:.4f}% de ausentismo")
    
    # Guardar modelo
    model_path = os.path.join(models_dir, 'absenteeism_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Modelo guardado en: {model_path}")
    
    # Guardar configuración del modelo en metadata para el API
    model_config = {
        'lookback': lookback,
        'horizon': horizon,
        'feature_size': X.shape[1]
    }
    with open(os.path.join(models_dir, 'absenteeism_config.pkl'), 'wb') as f:
        pickle.dump(model_config, f)
        
    print("Configuración de ausentismo guardada en models/absenteeism_config.pkl")

if __name__ == '__main__':
    train_model()
