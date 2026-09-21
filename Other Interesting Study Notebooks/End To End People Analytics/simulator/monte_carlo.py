import numpy as np
import pandas as pd

class MonteCarloSimulator:
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)
        
    def simulate_roi(self, employees_list, global_intervention_cost, effectiveness, n_simulations=5000):
        """
        Simula los escenarios de costos financieros con y sin intervención.
        
        Parámetros:
        - employees_list: Lista de diccionarios que representan a los empleados de alto riesgo.
                          Debe contener: 'role', 'monthly_salary', 'flight_risk_prob'.
        - global_intervention_cost: Costo fijo por empleado de aplicar la intervención (en MXN).
        - effectiveness: % de reducción de la probabilidad de fuga (e.g., 0.60 significa reducir la probabilidad en un 60%).
        - n_simulations: Número de iteraciones de Monte Carlo.
        
        Retorna:
        - Un diccionario con métricas agregadas y un array de ahorros netos para graficar la distribución.
        """
        n_emp = len(employees_list)
        if n_emp == 0:
            return {
                'avg_savings': 0.0,
                'p5_savings': 0.0,
                'p95_savings': 0.0,
                'prob_positive_roi': 0.0,
                'total_intervention_cost': 0.0,
                'expected_roi_pct': 0.0,
                'sim_results': np.zeros(n_simulations)
            }
            
        # Costo total de intervención
        total_intervention_cost = n_emp * global_intervention_cost
        
        # Estructurar parámetros de costo por empleado según su rol
        # Para Ventas Técnicas: Reemplazo promedio de 6.5 meses de salario (2x reclutamiento, 1.5x onboarding, 3x pérdida ventas)
        # Para Almacén: Reemplazo promedio de 2.5 meses de salario (1.2x reclutamiento, 0.8x onboarding, 0.5x productividad)
        emp_params = []
        for emp in employees_list:
            role = emp['role']
            salary = emp['monthly_salary']
            prob = emp['flight_risk_prob']
            
            if role == 'Ventas Técnicas':
                # Dist. normal para costos multiplicadores
                rec_mult = (1.5, 0.3)  # media, desv_est
                onb_mult = (1.5, 0.2)
                prod_mult = (3.0, 0.8)
            else: # Almacén
                rec_mult = (1.0, 0.15)
                onb_mult = (0.8, 0.1)
                prod_mult = (0.5, 0.1)
                
            emp_params.append({
                'salary': salary,
                'prob': prob,
                'rec_mult': rec_mult,
                'onb_mult': onb_mult,
                'prod_mult': prod_mult
            })
            
        sim_savings = []
        
        # Ejecutar la simulación de Monte Carlo
        for _ in range(n_simulations):
            cost_no_interv = 0.0
            cost_with_interv = 0.0
            
            for params in emp_params:
                sal = params['salary']
                p_quit = params['prob']
                
                # Muestrear costos de reemplazo aleatorios para esta iteración
                rec_cost = max(0.0, self.rng.normal(params['rec_mult'][0], params['rec_mult'][1])) * sal
                onb_cost = max(0.0, self.rng.normal(params['onb_mult'][0], params['onb_mult'][1])) * sal
                prod_cost = max(0.0, self.rng.normal(params['prod_mult'][0], params['prod_mult'][1])) * sal
                
                replacement_cost = rec_cost + onb_cost + prod_cost
                
                # --- Escenario A: Sin Intervención ---
                # Determinar si el empleado renuncia
                quit_no_interv = self.rng.random() < p_quit
                if quit_no_interv:
                    cost_no_interv += replacement_cost
                    
                # --- Escenario B: Con Intervención ---
                # La intervención cuesta un monto fijo
                cost_with_interv += global_intervention_cost
                
                # Probabilidad mitigada de fuga
                p_quit_mitigated = p_quit * (1.0 - effectiveness)
                quit_with_interv = self.rng.random() < p_quit_mitigated
                
                if quit_with_interv:
                    # Renuncia a pesar de la intervención
                    cost_with_interv += replacement_cost
                    
            # Ahorro neto en esta simulación
            saving = cost_no_interv - cost_with_interv
            sim_savings.append(saving)
            
        sim_savings = np.array(sim_savings)
        
        # Calcular métricas
        avg_savings = float(np.mean(sim_savings))
        p5 = float(np.percentile(sim_savings, 5))
        p95 = float(np.percentile(sim_savings, 95))
        
        prob_positive = float(np.mean(sim_savings > 0) * 100)
        
        expected_roi = 0.0
        if total_intervention_cost > 0:
            expected_roi = (avg_savings / total_intervention_cost) * 100
            
        return {
            'avg_savings': round(avg_savings, 2),
            'p5_savings': round(p5, 2),
            'p95_savings': round(p95, 2),
            'prob_positive_roi': round(prob_positive, 2),
            'total_intervention_cost': round(total_intervention_cost, 2),
            'expected_roi_pct': round(expected_roi, 2),
            'sim_results': sim_savings
        }
