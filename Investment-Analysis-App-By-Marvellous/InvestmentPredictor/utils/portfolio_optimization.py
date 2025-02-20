
import numpy as np
from scipy.optimize import minimize
import streamlit as st

def optimize_portfolio(returns, risk_tolerance=0.5):
    """Optimize portfolio weights using Modern Portfolio Theory"""
    try:
        n = returns.shape[1]
        
        # Calculate mean returns and covariance
        mu = returns.mean()
        Sigma = returns.cov()
        
        # Define optimization objective
        def objective(w):
            port_return = np.sum(mu * w)
            port_risk = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
            return -(risk_tolerance * port_return - (1-risk_tolerance) * port_risk)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        # Initial guess
        w0 = np.array([1./n for _ in range(n)])
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        return result.x if result.success else None
        
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return None
