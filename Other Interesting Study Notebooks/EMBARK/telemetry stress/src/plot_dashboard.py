import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_tactical_dashboard():
    # 1. Configuración de estilos retro-futuristas / terminal de control
    plt.rcParams['text.color'] = '#A3B8CC'
    plt.rcParams['axes.labelcolor'] = '#A3B8CC'
    plt.rcParams['xtick.color'] = '#6272A4'
    plt.rcParams['ytick.color'] = '#6272A4'
    plt.rcParams['font.family'] = 'monospace'

    # Cargar la telemetría procesada
    df = pd.read_csv('data/telemetry_stress_analysis.csv')

    # 2. Inicialización del canvas multi-panel (evitando plt.figure)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor='#0B0E14', sharex=True)

    # Paleta de colores "Cassette Futurism"
    color_bg = '#0B0E14'
    color_panel = '#141923'
    color_calm = '#00FF99'       # Verde Neón
    color_stress = '#FF3366'      # Rosa/Rojo Alerta
    color_grid = '#222A36'
    color_health = '#FFB86C'      # Naranja Vitals
    color_purple = '#BD93F9'

    # Segmentar fases para aplicar colores diferenciales
    calm_df = df[df['timestamp'] < 35]
    stress_df = df[df['timestamp'] >= 35]

    # --- PANEL 1: FEED BIOMÉTRICO VIRTUAL (Ritmo Cardíaco Estimado) ---
    ax1 = axes[0]
    ax1.set_facecolor(color_panel)
    ax1.plot(calm_df['timestamp'], calm_df['heart_rate_approx'], color=color_calm, lw=1.5, label='Calm Baseline')
    ax1.plot(stress_df['timestamp'], stress_df['heart_rate_approx'], color=color_stress, lw=1.5, label='Panic/Stress Peak')
    ax1.axvline(x=35, color='#FFCC00', linestyle='--', alpha=0.7, lw=2)
    ax1.text(35.5, 170, '▲ AMBUSH DETECTED', color='#FFCC00', fontsize=10, weight='bold')
    ax1.set_ylabel('VIRTUAL HR (BPM)', fontsize=11, weight='bold')
    ax1.grid(True, color=color_grid, linestyle=':')
    ax1.legend(loc='upper left', framealpha=0.1, facecolor=color_bg)
    ax1.set_title('TACTICAL NEURAL ANALYTICS // SESSION_REPLAY_042', color='#F8F8F2', fontsize=14, weight='bold', loc='left', pad=15)

    # --- PANEL 2: MOTOR TELEMETRY (Jitter del Ratón e Intensidad de Inputs) ---
    ax2 = axes[1]
    ax2.set_facecolor(color_panel)
    ax2.plot(calm_df['timestamp'], calm_df['kinematic_jitter'], color='#00CCFF', lw=1.5, label='Kinematic Jitter (Mouse Tremor)')
    ax2.plot(stress_df['timestamp'], stress_df['kinematic_jitter'], color='#FF66CC', lw=1.5)
    ax2.axvline(x=35, color='#FFCC00', linestyle='--', alpha=0.7, lw=2)
    ax2.set_ylabel('KINEMATIC JITTER', fontsize=11, weight='bold')
    ax2.grid(True, color=color_grid, linestyle=':')
    ax2.legend(loc='upper left', framealpha=0.1, facecolor=color_bg)

    # Doble eje Y para la densidad de clicks (Panic Click Index)
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(df['timestamp'], df['panic_click_index'], color=color_purple, alpha=0.15)
    ax2_twin.set_ylabel('INPUT DENSITY (ACTIONS/SEC)', color=color_purple, fontsize=10, weight='bold')
    ax2_twin.tick_params(axis='y', labelcolor=color_purple)

    # --- PANEL 3: ESTADO DE SALUD (Vitals del Jugador) ---
    ax3 = axes[2]
    ax3.set_facecolor(color_panel)
    ax3.plot(df['timestamp'], df['player_health'], color=color_health, lw=2, label='Player Health %')
    ax3.fill_between(df['timestamp'], df['player_health'], 0, color=color_health, alpha=0.05)
    ax3.axvline(x=35, color='#FFCC00', linestyle='--', alpha=0.7, lw=2)
    ax3.set_xlabel('SESSION TIME (SECONDS)', fontsize=11, weight='bold', labelpad=10)
    ax3.set_ylabel('VITALS (%)', fontsize=11, weight='bold')
    ax3.set_ylim(0, 110)
    ax3.grid(True, color=color_grid, linestyle=':')
    ax3.legend(loc='lower left', framealpha=0.1, facecolor=color_bg)

    # Estilizar bordes globales de los paneles
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('#282A36')
        ax.tick_params(colors='#6272A4')

    plt.tight_layout()
    
    # Guardar imagen final
    output_filename = 'reports/figures/tactical_neural_analytics.png'
    plt.savefig(output_filename, dpi=150, facecolor=color_bg)
    print(f"Dashboard guardado exitosamente como: {output_filename}")

if __name__ == "__main__":
    generate_tactical_dashboard()