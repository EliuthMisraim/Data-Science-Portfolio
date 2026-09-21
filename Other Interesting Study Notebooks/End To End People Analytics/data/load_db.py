import os
import sqlite3
import pandas as pd

def load_database():
    db_path = 'data/people_analytics.db'
    schema_path = 'data/schema.sql'
    
    print(f"Conectando a la base de datos SQLite en: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Ejecutar el esquema SQL para crear las tablas
    print(f"Cargando el esquema SQL desde {schema_path}...")
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    cursor.executescript(schema_sql)
    conn.commit()
    print("Esquema de base de datos cargado exitosamente.")
    
    # 2. Cargar los datos desde los archivos CSV
    csv_tables = {
        'employees': 'data/employees.csv',
        'nom035_survey': 'data/nom035_survey.csv',
        'performance_sales': 'data/performance_sales.csv',
        'performance_warehouse': 'data/performance_warehouse.csv',
        'absenteeism_events': 'data/absenteeism_events.csv',
        'absenteeism_history': 'data/absenteeism_history.csv'
    }
    
    for table_name, csv_path in csv_tables.items():
        if os.path.exists(csv_path):
            print(f"Cargando datos en la tabla '{table_name}' desde {csv_path}...")
            # Leer CSV con pandas
            df = pd.read_csv(csv_path)
            
            # Limpiar tabla antes de insertar
            cursor.execute(f"DELETE FROM {table_name}")
            
            # Insertar en base de datos
            df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"Insertados {len(df)} registros en la tabla '{table_name}'.")
        else:
            print(f"ADVERTENCIA: No se encontró el archivo {csv_path}. Saltando la tabla '{table_name}'.")
            
    conn.commit()
    
    # 3. Validar carga haciendo algunas consultas básicas
    print("\n--- Validando Integridad de la Base de Datos ---")
    
    cursor.execute("SELECT COUNT(*) FROM employees")
    num_emp = cursor.fetchone()[0]
    print(f"Total empleados en DB: {num_emp}")
    
    cursor.execute("SELECT role, COUNT(*), ROUND(AVG(monthly_salary), 2) FROM employees GROUP BY role")
    print("Empleados por Rol y Salario Promedio:")
    for row in cursor.fetchall():
        print(f"  - {row[0]}: {row[1]} colaboradores | Salario Promedio: ${row[2]} MXN")
        
    cursor.execute("SELECT COUNT(*) FROM employees WHERE has_quit = 1")
    num_quit = cursor.fetchone()[0]
    print(f"Total empleados que renunciaron (has_quit = 1): {num_quit} (Tasa de rotación simulada: {num_quit/num_emp*100:.1f}%)")
    
    cursor.execute("SELECT COUNT(*) FROM absenteeism_events")
    num_events = cursor.fetchone()[0]
    print(f"Total de eventos de ausentismo individuales: {num_events}")
    
    cursor.execute("SELECT COUNT(*) FROM absenteeism_history")
    num_hist = cursor.fetchone()[0]
    print(f"Total de registros históricos de series de tiempo (Monterrey CDIX): {num_hist}")
    
    conn.close()
    print("\nProceso de carga completado exitosamente y base de datos lista.")

if __name__ == '__main__':
    load_database()
