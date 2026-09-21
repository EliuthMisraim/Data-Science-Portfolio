import os
import pickle
import math
import numpy as np
import pandas as pd

class ModelManager:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.flight_risk_model = None
        self.flight_risk_explainer = None
        self.metadata = None
        
        self.absenteeism_model = None
        self.absenteeism_scaler = None
        self.absenteeism_config = None

        self.retention_model = None       # RandomForest de recomendación de retención
        self.retention_label_encoder = None
        self.retention_feature_cols = None
        
        self.load_models()
        
    def load_models(self):
        # 1. Cargar modelo Flight Risk
        fr_model_path = os.path.join(self.models_dir, 'flight_risk_model.pkl')
        fr_expl_path = os.path.join(self.models_dir, 'flight_risk_explainer.pkl')
        meta_path = os.path.join(self.models_dir, 'metadata.pkl')
        
        if os.path.exists(fr_model_path) and os.path.exists(fr_expl_path) and os.path.exists(meta_path):
            print("Cargando modelo de riesgo de deserción y explicador SHAP...")
            with open(fr_model_path, 'rb') as f:
                self.flight_risk_model = pickle.load(f)
            with open(fr_expl_path, 'rb') as f:
                self.flight_risk_explainer = pickle.load(f)
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            print("ADVERTENCIA: Archivos del modelo de Flight Risk no encontrados. ¿Ya se entrenó?")
            
        # 2. Cargar modelo de ausentismo MLP (Red Neuronal)
        ab_model_path = os.path.join(self.models_dir, 'absenteeism_model.pkl')
        ab_scaler_path = os.path.join(self.models_dir, 'absenteeism_scaler.pkl')
        ab_config_path = os.path.join(self.models_dir, 'absenteeism_config.pkl')
        
        if os.path.exists(ab_model_path) and os.path.exists(ab_scaler_path) and os.path.exists(ab_config_path):
            print("Cargando modelo de ausentismo (Red Neuronal) y configuraciones...")
            with open(ab_model_path, 'rb') as f:
                self.absenteeism_model = pickle.load(f)
            with open(ab_scaler_path, 'rb') as f:
                self.absenteeism_scaler = pickle.load(f)
            with open(ab_config_path, 'rb') as f:
                self.absenteeism_config = pickle.load(f)
        else:
            print("ADVERTENCIA: Archivos del modelo de ausentismo no encontrados. ¿Ya se entrenó?")

        # 3. Cargar modelo de retención (recomendación de intervención)
        ret_model_path = os.path.join(self.models_dir, 'retention_model.pkl')
        if os.path.exists(ret_model_path):
            print("Cargando modelo de retención (RandomForest de intervención)...")
            with open(ret_model_path, 'rb') as f:
                ret_artifact = pickle.load(f)
            self.retention_model = ret_artifact['model']
            self.retention_label_encoder = ret_artifact['label_encoder']
            self.retention_feature_cols = ret_artifact['feature_cols']
        else:
            print("ADVERTENCIA: retention_model.pkl no encontrado. Ejecuta models/retention_model.py.")

    def predict_flight_risk(self, req_data):
        """
        Ejecuta la predicción de Flight Risk para un empleado y calcula sus valores SHAP.
        """
        if self.flight_risk_model is None or self.flight_risk_explainer is None:
            raise ValueError("El modelo de Flight Risk no está cargado.")
            
        # Codificación
        role_encoded = 1 if req_data.role == 'Ventas Técnicas' else 0
        gender_encoded = 0
        if req_data.gender == 'Femenino':
            gender_encoded = 1
        elif req_data.gender == 'No binario':
            gender_encoded = 2
            
        # Armar vector de características en el orden correcto
        feature_vector = [
            role_encoded, req_data.age, gender_encoded, req_data.tenure_months, req_data.monthly_salary,
            req_data.distance_km, req_data.overtime_hours_last_month, req_data.training_hours, req_data.last_performance_rating,
            req_data.leadership_index, req_data.workload_index, req_data.organizational_environment_index,
            req_data.quota_attainment_pct, req_data.sales_closed, req_data.commissions_earned,
            req_data.on_time_delivery_pct, req_data.inventory_errors, req_data.machinery_incidents,
            req_data.bradford_factor
        ]
        
        # DataFrame de una sola fila
        df_features = pd.DataFrame([feature_vector], columns=self.metadata['feature_cols'])
        
        # Predicción probabilística
        prob = float(self.flight_risk_model.predict_proba(df_features)[0, 1])
        high_risk = prob >= 0.50
        
        # Calcular valores SHAP individuales
        shap_out = self.flight_risk_explainer(df_features)
        
        # Extraer valores y mapear con los nombres de características
        if hasattr(shap_out, 'values'):
            vals = shap_out.values[0]
            # Si SHAP retorna estructura tridimensional (para multiclase), manejamos la clase 1
            if len(vals.shape) > 1 and vals.shape[-1] == 2:
                vals = vals[:, 1]
        else:
            # Fallback a la forma antigua
            vals = self.flight_risk_explainer.shap_values(df_features)
            if isinstance(vals, list) and len(vals) == 2:
                vals = vals[1][0]
            elif isinstance(vals, np.ndarray) and len(vals.shape) == 3: # (samples, features, classes)
                vals = vals[0, :, 1]
            else:
                vals = vals[0]
                
        vals = [float(v) for v in vals]
        
        # Mapear nombres amigables
        shap_dict = {}
        for col, val in zip(self.metadata['feature_cols'], vals):
            friendly_name = {
                'role_encoded': 'Rol (Ventas vs Almacén)',
                'age': 'Edad',
                'gender_encoded': 'Género',
                'tenure_months': 'Antigüedad (Meses)',
                'monthly_salary': 'Salario Mensual',
                'distance_km': 'Distancia a Oficina (km)',
                'overtime_hours_last_month': 'Horas Extra de Último Mes',
                'training_hours': 'Horas de Capacitación',
                'last_performance_rating': 'Evaluación de Desempeño anterior',
                'leadership_index': 'NOM-035: Índice de Liderazgo',
                'workload_index': 'NOM-035: Carga de Trabajo/Estrés',
                'organizational_environment_index': 'NOM-035: Entorno Organizacional',
                'quota_attainment_pct': 'Cumplimiento de Cuota Ventas',
                'sales_closed': 'Ventas Cerradas',
                'commissions_earned': 'Comisiones Ganadas',
                'on_time_delivery_pct': 'Entregas a Tiempo Almacén',
                'inventory_errors': 'Errores de Inventario',
                'machinery_incidents': 'Incidentes con Maquinaria',
                'bradford_factor': 'Índice de Bradford (Ausentismo)'
            }.get(col, col)
            
            # Filtramos características no relevantes del rol
            if col in ['quota_attainment_pct', 'sales_closed', 'commissions_earned'] and req_data.role == 'Almacén':
                continue
            if col in ['on_time_delivery_pct', 'inventory_errors', 'machinery_incidents'] and req_data.role == 'Ventas Técnicas':
                continue
                
            shap_dict[friendly_name] = round(val, 4)
            
        return prob, high_risk, shap_dict

    def predict_retention(self, row: dict) -> dict:
        """
        Predice la intervención de retención recomendada para un empleado.

        Parámetros
        ----------
        row : dict
            Diccionario con los datos del empleado (mismas claves que las features del modelo).

        Retorna
        -------
        dict con:
            - recommended_action: str — la intervención con mayor probabilidad
            - action_probabilities: dict {accion: probabilidad}
        """
        if self.retention_model is None:
            raise ValueError("El modelo de retención no está cargado.")

        # Calcular percentil de salario estimado (aproximado con escala fija)
        # En producción esto vendría de la BD; aquí usamos un valor neutro
        row_with_percentile = dict(row)
        if 'salary_percentile' not in row_with_percentile:
            row_with_percentile['salary_percentile'] = 50.0

        X = pd.DataFrame([[row_with_percentile.get(col, 0.0) for col in self.retention_feature_cols]],
                         columns=self.retention_feature_cols)

        pred_enc = self.retention_model.predict(X)[0]
        proba = self.retention_model.predict_proba(X)[0]
        classes = self.retention_label_encoder.classes_

        recommended_action = self.retention_label_encoder.inverse_transform([pred_enc])[0]
        action_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        return {
            'recommended_action': recommended_action,
            'action_probabilities': action_probs
        }

    def forecast_absenteeism(self, req_data):
        """
        Genera pronóstico de ausentismo para las próximas 4 semanas usando el modelo MLP.
        """
        if self.absenteeism_model is None or self.absenteeism_scaler is None:
            raise ValueError("El modelo de ausentismo no está cargado.")
            
        dept_encoded = 0 if req_data.department == 'Almacén' else 1
        
        # 1. Escalar la tasa de ausentismo histórica
        r_min = self.absenteeism_scaler['rate_min']
        r_max = self.absenteeism_scaler['rate_max']
        
        hist_scaled = [(r - r_min) / (r_max - r_min) for r in req_data.historical_rates]
        
        # 2. Construir secuencia de características para los 8 pasos históricos
        sins = []
        coss = []
        
        # Determinar semana de inicio
        start_week = (req_data.target_week_of_year - 8) % 52
        if start_week == 0:
            start_week = 52
            
        for idx in range(8):
            w_of_yr = (start_week + idx) % 52
            if w_of_yr == 0:
                w_of_yr = 52
                
            sins.append(math.sin(2 * math.pi * w_of_yr / 52.0))
            coss.append(math.cos(2 * math.pi * w_of_yr / 52.0))
            
        # Aplanar vector de características: rates + sins + coss + dept_encoded
        features = np.concatenate([hist_scaled, sins, coss, [dept_encoded]])
        
        # Reshape para hacer inferencia en 2D [1, n_features]
        features_2d = features.reshape(1, -1)
        
        # 3. Predicción
        output_scaled = self.absenteeism_model.predict(features_2d)[0]
        
        # 4. Des-escalar la predicción
        forecasted_rates = [float(o * (r_max - r_min) + r_min) for o in output_scaled]
        forecasted_rates = [max(0.0, min(1.0, r)) for r in forecasted_rates]
        
        return forecasted_rates
