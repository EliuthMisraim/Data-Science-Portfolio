import pandas as pd
import sys
import io

# Asegurar codificación UTF-8 para stdout en Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def probar_pipeline_telemetria():
    print("="*50)
    print("INICIANDO VALIDACIÓN DE MÉTRICAS DE TELEMETRÍA")
    print("="*50)
    
    # 1. Cargar el archivo generado
    try:
        df = pd.read_csv('data/telemetry_stress_analysis.csv')
        print("✔ Archivo CSV cargado correctamente.")
    except FileNotFoundError:
        print("❌ Error: No se encontró 'data/telemetry_stress_analysis.csv'. Ejecuta primero el simulador.")
        return

    # 2. Separar los datos por las fases del diseño del juego
    # Fase 1: Exploración (0 a 35 segundos) | Fase 2: Emboscada (35 a 60 segundos)
    fase_calma = df[df['timestamp'] < 35]
    fase_estres = df[df['timestamp'] >= 35]

    # 3. Extraer métricas clave para comparar
    jitter_calma = fase_calma['kinematic_jitter'].mean()
    jitter_estres = fase_estres['kinematic_jitter'].mean()
    
    clicks_calma = fase_calma['panic_click_index'].mean()
    clicks_estres = fase_estres['panic_click_index'].mean()
    
    hr_calma = fase_calma['heart_rate_approx'].mean()
    hr_estres = fase_estres['heart_rate_approx'].mean()

    # 4. Asserts / Pruebas de Hipótesis Básicas
    print("\n--- Analizando Resultados Obtenidos ---")
    print(f"Fase Calma  -> Jitter: {jitter_calma:.2f} | Clicks/s: {clicks_calma:.2f} | Promedio HR: {hr_calma:.1f} BPM")
    print(f"Fase Estrés -> Jitter: {jitter_estres:.2f} | Clicks/s: {clicks_estres:.2f} | Promedio HR: {hr_estres:.1f} BPM")
    
    print("\n--- Evaluación del Sensor Virtual ---")
    
    # Test 1: El temblor del ratón (jitter) debe ser significativamente mayor en la emboscada
    if jitter_estres > (jitter_calma * 3):
        print("✔ TEST JITTER PASADO: El temblor cinemático aumentó correctamente durante la emboscada.")
    else:
        print("❌ TEST JITTER FALLIDO: El temblor no refleja una respuesta de pánico.")

    # Test 2: El ritmo cardíaco aproximado debe superar un umbral de estrés (ej. > 120 BPM)
    if hr_estres > 120 and hr_calma < 90:
        print("✔ TEST BIOMÉTRICO PASADO: El 'Virtual Heart Rate' discrimina con éxito los dos estados fisiológicos.")
    else:
        print("❌ TEST BIOMÉTRICO FALLIDO: Los umbrales de ritmo cardíaco no son realistas.")
        
    print("="*50)

if __name__ == "__main__":
    probar_pipeline_telemetria()