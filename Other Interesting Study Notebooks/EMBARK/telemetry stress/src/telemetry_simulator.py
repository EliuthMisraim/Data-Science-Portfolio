import numpy as np
import pandas as pd

def generate_virtual_biometric_telemetry(duration_sec=60, sampling_rate_hz=30):
    """
    Simula telemetría cruda de juego y aplica ingeniería de características
    para estimar el estrés del jugador (Sensor Biométrico Virtual).
    """
    np.random.seed(42)
    fs = sampling_rate_hz
    n_samples = fs * duration_sec
    timestamps = np.linspace(0, duration_sec, n_samples)

    # Inicialización de vectores
    delta_x = np.zeros(n_samples)
    delta_y = np.zeros(n_samples)
    actions = np.zeros(n_samples)
    health = np.ones(n_samples) * 100

    # Definimos el punto de quiebre: Emboscada a los 35 segundos
    calm_samples = int(35 * fs)
    stress_samples = n_samples - calm_samples

    # --- FASE 1: Exploración Calmada (0 - 35s) ---
    # Movimientos amplios y limpios (baja frecuencia, ruido controlado)
    delta_x[:calm_samples] = np.random.normal(0, 2.0, calm_samples) + np.sin(timestamps[:calm_samples] * 0.5) * 5
    delta_y[:calm_samples] = np.random.normal(0, 1.5, calm_samples)
    # Acciones espaciadas (p. ej., lootear de forma calmada)
    actions[:calm_samples] = np.random.choice([0, 1], size=calm_samples, p=[0.95, 0.05])

    # --- FASE 2: Emboscada y Pánico Técnico (35s - 60s) ---
    # Micro-temblores (Jitter cinemático alto por adrenalina) y movimientos erráticos
    delta_x[calm_samples:] = np.random.normal(0, 12.0, stress_samples) + np.sin(timestamps[calm_samples:] * 5.0) * 15
    delta_y[calm_samples:] = np.random.normal(0, 10.0, stress_samples) + np.cos(timestamps[calm_samples:] * 5.0) * 10
    # Spam de botones (Panic Clicking / Inputs ineficientes)
    actions[calm_samples:] = np.random.choice([0, 1, 2], size=stress_samples, p=[0.60, 0.30, 0.10])
    # Pérdida progresiva de salud
    health_drop = np.linspace(100, 15, stress_samples) + np.random.normal(0, 2, stress_samples)
    health[calm_samples:] = np.clip(health_drop, 10, 100)

    # Dataframe de telemetría cruda
    df = pd.DataFrame({
        'timestamp': timestamps,
        'delta_x': delta_x,
        'delta_y': delta_y,
        'action_triggered': actions,
        'player_health': health
    })

    # ==========================================
    # FEATURE ENGINEERING (PROXIES DE ESTRÉS)
    # ==========================================
    
    # 1. Magnitud del vector de velocidad del ratón
    df['mouse_velocity'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    
    # 2. Aceleración absoluta (Derivada de la velocidad)
    df['mouse_acceleration'] = df['mouse_velocity'].diff().fillna(0).abs()

    # 3. Jitter Cinemático: Desviación estándar de la aceleración en una ventana móvil de 1 segundo
    window_size = fs * 1  # 30 muestras
    df['kinematic_jitter'] = df['mouse_acceleration'].rolling(window=window_size, min_periods=1).std()

    # 4. Índice de Click de Pánico: Densidad de inputs en la ventana de 1 segundo
    df['panic_click_index'] = df['action_triggered'].rolling(window=window_size, min_periods=1).sum()

    # ==========================================
    # CÁLCULO DEL VIRTUAL HEART RATE (Proxy target)
    # ==========================================
    # Línea base de 70 BPM. El estrés cinemático, el spam de botones y la vida baja inyectan pulsaciones.
    baseline_hr = 70
    jitter_contrib = (df['kinematic_jitter'] / df['kinematic_jitter'].max()) * 60
    click_contrib = (df['panic_click_index'] / df['panic_click_index'].max()) * 30
    health_contrib = ((100 - df['player_health']) / 100) * 30

    df['heart_rate_approx'] = baseline_hr + jitter_contrib + click_contrib + health_contrib
    
    # Añadir ruido fisiológico de alta frecuencia y suavizar con una media móvil corta
    df['heart_rate_approx'] += np.random.normal(0, 1.5, n_samples)
    df['heart_rate_approx'] = df['heart_rate_approx'].rolling(window=5, min_periods=1).mean()
    
    # Rellenar valores nulos iniciales causados por las ventanas móviles
    df = df.bfill()

    return df

if __name__ == "__main__":
    # Ejecutar procesamiento
    telemetry_df = generate_virtual_biometric_telemetry()
    
    # Guardar los datos procesados listos para visualización o entrenamiento de modelos
    telemetry_df.to_csv('data/telemetry_stress_analysis.csv', index=False)
    print("¡Procesamiento de telemetría completado con éxito!")