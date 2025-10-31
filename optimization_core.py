import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge 
from scipy.optimize import minimize
import data_processor as dp 

# =========================================================================
# 1. FEATURE ENGINEERING AND MODEL TRAINING (Log-NNLS)
# =========================================================================

def transform_features(X_data):
    X_transformed = pd.DataFrame()
    for col in X_data.columns:
        X_transformed[f'Log_{col}'] = np.log1p(X_data[col])
        
    return X_transformed

def train_sales_model(X_transformed, y):
    model = Ridge(alpha=0.1, fit_intercept=True, positive=True)
    model.fit(X_transformed, y)
    
    # Store model outputs globally in data_processor namespace for report access
    dp.GLOBAL_COEF_DICT = dict(zip(X_transformed.columns, model.coef_))
    dp.GLOBAL_INTERCEPT = model.intercept_
    dp.MODEL_SCORE = model.score(X_transformed, y)
    
    print(f"✅ Predictive Log-NNLS Model Trained (R-squared: {dp.MODEL_SCORE:.4f})")
    
    return dp.GLOBAL_COEF_DICT, dp.GLOBAL_INTERCEPT

# =========================================================================
# 2. OPTIMIZATION CORE (Maximize Revenue subject to Budget)
# =========================================================================

def run_budget_optimization(coef_dict, intercept, total_budget):
    
    log_cols = [f'Log_{c}' for c in dp.GLOBAL_CHANNEL_NAMES]
    coefficients = np.array([coef_dict[c] for c in log_cols])

    # Objective function: Minimize (-Incremental Revenue)
    def objective(spend_by_channel):
        """
        Calculates predicted incremental Revenue using the Log-NNLS Model.
        The intercept (baseline cost/loss) is intentionally removed here.
        """
        log_transformed_spend = np.log1p(spend_by_channel)
        
        # Predicted Revenue = sum(Coefficient * Log_Spend). Intercept is REMOVED.
        predicted_revenue = (coefficients * log_transformed_spend).sum() 
        
        return -predicted_revenue
    
    # Constraint function: Total spend must be <= TOTAL_BUDGET
    def constraint_budget(spend_by_channel):
        return total_budget - np.sum(spend_by_channel)

    constraints = ({'type': 'ineq', 'fun': constraint_budget})

    # Bounds: Spend must be non-negative (>= 0)
    bounds = [(0, total_budget) for _ in dp.GLOBAL_CHANNEL_NAMES]

    # Initial Guess: Even split of the budget
    initial_guess = np.array([total_budget / len(dp.GLOBAL_CHANNEL_NAMES)] * len(dp.GLOBAL_CHANNEL_NAMES))

    # Run the optimization
    result = minimize(
        objective, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'disp': False}
    )
    
    optimal_spend = result.x
    optimal_revenue = -result.fun
    
    return optimal_spend, optimal_revenue, result.success

# =========================================================================
# 3. MARGINAL ROI CALCULATOR
# =========================================================================

def calculate_marginal_roi(channel_name, base_spend=100.0):
    """Calculates Marginal ROI (MROI) Effectiveness for benchmarking."""
    log_coef_name = f'Log_{channel_name}'
    
    if log_coef_name in dp.GLOBAL_COEF_DICT:
        coefficient = dp.GLOBAL_COEF_DICT[log_coef_name]
        mroi_effectiveness = coefficient / (1 + base_spend) 
        return mroi_effectiveness
    return 0.0