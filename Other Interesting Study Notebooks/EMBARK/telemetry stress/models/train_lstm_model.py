import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

def run_macro_churn_pipeline():
    np.random.seed(42)
    
    # =========================================================================
    # 1. SIMULACIÓN DE DATOS MACRO (TENSOR 3D: 5000 Jugadores, 10 Juegos, 5 Features)
    # =========================================================================
    n_players = 5000
    lookback_games = 10
    n_features = 5  # [avg_jitter, panic_clicks, win_rate, final_health, match_duration]

    X = np.zeros((n_players, lookback_games, n_features))
    y = np.zeros((n_players, 1))

    for i in range(n_players):
        # Tasa de Churn histórica simulada del 30%
        will_churn = np.random.rand() < 0.30
        
        # Comportamiento base estable (Jugador saludable que se queda)
        jitter = np.random.normal(1.5, 0.3, lookback_games)
        clicks = np.random.normal(2.0, 0.5, lookback_games)
        win_rate = np.random.choice([0, 1], size=lookback_games, p=[0.6, 0.4])
        health = np.random.normal(60, 15, lookback_games)
        duration = np.random.normal(20, 4, lookback_games) # en minutos
        
        if will_churn:
            # Trayectoria de "Rage Quit" en las últimas 4 partidas de su historial
            jitter[-4:] += np.linspace(1.0, 6.0, 4) + np.random.normal(0, 0.5, 4)
            clicks[-4:] += np.linspace(2.0, 12.0, 4)
            win_rate[-4:] = 0  # Racha severa de derrotas
            health[-4:] = np.clip(np.linspace(50, 10, 4) + np.random.normal(0, 5, 4), 0, 100)
            duration[-4:] = np.linspace(20, 5, 4) # Sesiones cortas (alt+f4 o abandono temprano)
            y[i] = 1

        # Poblar el espacio del tensor para el jugador i
        X[i, :, 0] = np.clip(jitter, 0, 15)
        X[i, :, 1] = np.clip(clicks, 0, 30)
        X[i, :, 2] = win_rate
        X[i, :, 3] = np.clip(health, 0, 100)
        X[i, :, 4] = np.clip(duration, 0, 60)

    # Normalización estadística Z-score por Feature para optimización del Gradiente
    for f in [0, 1, 3, 4]:
        mean, std = X[:, :, f].mean(), X[:, :, f].std()
        X[:, :, f] = (X[:, :, f] - mean) / (std + 1e-8)

    # División de datos: 80% Entrenamiento, 20% Validación
    split_idx = int(n_players * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # =========================================================================
    # 2. COMPILACIÓN DE ARQUITECTURA LSTM (KERAS 3)
    # =========================================================================
    model = keras.Sequential([
        layers.Input(shape=(lookback_games, n_features)),
        layers.LSTM(64, return_sequences=False, activation='tanh'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    # =========================================================================
    # 3. ENTRENAMIENTO DEL MODELO
    # =========================================================================
    epochs = 15
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        verbose=1
    )

    # =========================================================================
    # 4. VISUALIZACIÓN DE PERFORMANCE (DIRECCIÓN DE ARTE TÁCTICA)
    # =========================================================================
    plt.rcParams['text.color'] = '#A3B8CC'
    plt.rcParams['axes.labelcolor'] = '#A3B8CC'
    plt.rcParams['font.family'] = 'monospace'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0B0E14')

    # Panel Izquierdo: Decaimiento de Pérdida (Loss)
    ax1.set_facecolor('#141923')
    ax1.plot(history.history['loss'], color='#FF3366', lw=2, label='Train Loss')
    ax1.plot(history.history['val_loss'], color='#FFB86C', lw=2, linestyle='--', label='Val Loss')
    ax1.set_title('MODEL LOSS DECAY // OPT_BINARY_CROSSENTROPY', color='#F8F8F2', fontsize=12, weight='bold', loc='left')
    ax1.set_xlabel('EPOCH', weight='bold')
    ax1.set_ylabel('LOSS', weight='bold')
    ax1.grid(True, color='#222A36', linestyle=':')
    ax1.legend(framealpha=0.1, facecolor='#0B0E14')

    # Panel Derecho: Métrica de Negocio (AUC Core)
    ax2.set_facecolor('#141923')
    ax2.plot(history.history['auc'], color='#00FF99', lw=2, label='Train AUC')
    ax2.plot(history.history['val_auc'], color='#00CCFF', lw=2, linestyle='--', label='Val AUC')
    ax2.set_title('AREA UNDER CURVE (AUC) // PERFORMANCE_METRIC', color='#F8F8F2', fontsize=12, weight='bold', loc='left')
    ax2.set_xlabel('EPOCH', weight='bold')
    ax2.set_ylabel('AUC SCORE', weight='bold')
    ax2.grid(True, color='#222A36', linestyle=':')
    ax2.legend(framealpha=0.1, facecolor='#0B0E14')

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('#282A36')
        ax.tick_params(colors='#6272A4')

    plt.tight_layout()
    plt.savefig('reports/figures/lstm_training_performance.png', dpi=150, facecolor='#0B0E14')
    print("Métricas guardadas exitosamente como 'reports/figures/lstm_training_performance.png'")

if __name__ == "__main__":
    run_macro_churn_pipeline()