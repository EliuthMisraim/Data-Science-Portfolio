import os
import sys
import subprocess
import time
import signal

def run_step(command_args, description):
    print(f"\n--- Ejecutando: {description} ---")
    print(f"Comando: {' '.join(command_args)}")
    process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Imprimir salida en tiempo real
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    if rc != 0:
        print(f"[ERROR] Al ejecutar '{description}' (codigo de salida: {rc})")
        sys.exit(rc)
    print(f"[OK] Completado: {description}")

def main():
    # Rutas clave
    db_path = os.path.join('data', 'people_analytics.db')
    model_risk_path = os.path.join('models', 'flight_risk_model.pkl')
    model_abs_path = os.path.join('models', 'absenteeism_model.pkl')
    
    # 1. Comprobar e Inicializar Datos si es necesario
    if not os.path.exists(db_path):
        print("Base de datos no encontrada. Generando datos ficticios y cargando SQL...")
        run_step([sys.executable, os.path.join('data', 'generate_data.py')], "Generacion de Mock Data")
        run_step([sys.executable, os.path.join('data', 'load_db.py')], "Carga de Base de Datos SQLite")
    else:
        print("[OK] Base de datos detectada.")
        
    # 2. Comprobar y Entrenar Modelos si es necesario
    if not os.path.exists(model_risk_path) or not os.path.exists(model_abs_path):
        print("Modelos no entrenados. Iniciando fase de entrenamiento...")
        run_step([sys.executable, os.path.join('models', 'train_flight_risk.py')], "Entrenamiento de Flight Risk (XGBoost)")
        run_step([sys.executable, os.path.join('models', 'train_absenteeism.py')], "Entrenamiento de Ausentismo (Red Neuronal)")
    else:
        print("[OK] Modelos entrenados detectados.")
        
    # 3. Arrancar los Microservicios en Paralelo
    print("\n==============================================")
    print("[STARTING] Arrancando microservicios de People Analytics")
    print("==============================================")
    
    # Arrancar FastAPI
    api_cmd = [
        "uvicorn", 
        "api.main:app", 
        "--host", "127.0.0.1", 
        "--port", "8000"
    ]
    print(f"Iniciando API Backend en http://127.0.0.1:8000")
    api_process = subprocess.Popen(api_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Arrancar Streamlit
    app_cmd = [
        "streamlit", 
        "run", 
        os.path.join('app', 'main.py'), 
        "--server.port", "8501"
    ]
    print(f"Iniciando App Frontend en http://127.0.0.1:8501")
    app_process = subprocess.Popen(app_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Manejar Ctrl+C para apagar ambos procesos
    def signal_handler(sig, frame):
        print("\n[STOPPING] Apagando servicios...")
        app_process.terminate()
        api_process.terminate()
        print("Servicios apagados exitosamente.")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\nServicios iniciados. Presiona Ctrl+C para detenerlos.")
    
    # Monitorear salidas de procesos en el hilo principal
    try:
        while True:
            # Comprobar si algún proceso murió inesperadamente
            api_status = api_process.poll()
            app_status = app_process.poll()
            
            if api_status is not None:
                print(f"[WARN] El servidor API termino inesperadamente con codigo {api_status}")
                app_process.terminate()
                break
                
            if app_status is not None:
                print(f"[WARN] La aplicacion Streamlit termino inesperadamente con codigo {app_status}")
                api_process.terminate()
                break
                
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        app_process.terminate()
        api_process.terminate()

if __name__ == '__main__':
    main()
