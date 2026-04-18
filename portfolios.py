"""
PORTFOLIOS: FUNCIONES DE CONSTRUCCIÓN DE CARTERAS
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martínez
"""

import numpy as np
from metrics import shrinkage
from scipy.optimize import minimize
from hrp import HRP, HERC

def W_EW(K):
    """
    CARTERA EQUIPONDERADA (EW): 
    Mismos pesos para los K ETFs
    
    [Pesos]
        w_k = 1 / K
    """
    return np.full(K, 1.0 / K)


def W_VT(Sigma, K):
    """
    CARTERA VOLATILITY TIMING (VT):
    Penaliza utilizando la varianza
    
    [Pesos]
        w_{k,t} = (1 / σ^2_{k,t}) / Σ(1 / σ^2_{k,t})
    """
    variances = np.diag(Sigma)
    variances = np.where(variances > 0, variances, np.inf)
    
    inv_vol = 1.0 / variances
    inv_vol = np.where(np.isfinite(inv_vol), inv_vol, 0.0)

    sum_inv_vol = inv_vol.sum()
    
    return inv_vol / sum_inv_vol


def W_RRT(mu, Sigma, K):
    """
    CARTERA REWARD-TO-RISK TIMING (RRT):
    Incorpora la prima de riesgo (μ) y penaliza por varianza (σ^2)
    
    [Pesos]
        w_{k,t} ∝ μ^+_{k,t} / σ^2_{k,t}
        donde μ^+_{k,t} = max(μ_{k,t}, 0)
    """
    if K == 1:
        return np.array([1.0])

    mu_plus = np.maximum(mu, 0.0)
    
    variances = np.diag(Sigma)
    variances = np.where(variances > 0, variances, np.inf)
    
    ratio = mu_plus / variances
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    
    sum_ratio = ratio.sum()
    
    if sum_ratio == 0 or np.isnan(sum_ratio):
        return np.full(K, 1.0 / K)
    
    return ratio / sum_ratio


def W_GMV(R_IS: np.ndarray, K: int, print_delta: bool = False) -> np.ndarray:
    if K == 1:
        return np.array([1.0])

    Sigma_clean, delta = shrinkage(R_IS)
    if print_delta:
        print(f" Shrinkage Delta óptimo (GMV): {delta:.4f}")

    escala = 1.0 / np.mean(np.diag(Sigma_clean))
    Sigma_opt = Sigma_clean * escala

    def objective(w):
        return 0.5 * (w.T @ Sigma_opt @ w)

    def jacobiano(w):
        return Sigma_opt @ w

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = tuple((0.0, 1.0) for _ in range(K))
    w0 = np.full(K, 1.0 / K)

    res = minimize(
        fun=objective,
        x0=w0,
        method='SLSQP',
        jac=jacobiano,
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False}
    )

    if res.success:
        w_star = np.clip(res.x, 0.0, 1.0)
        suma_pesos = np.sum(w_star)
        if suma_pesos > 0.0:
            return w_star / suma_pesos

    return W_VT(Sigma_clean, K)

    
def W_ERC(Sigma, K):
    """
    CARTERA EQUAL RISK CONTRIBUTION (ERC).
    Busca que cada ETF contribuya igual al riesgo total de la cartera. (Roncalli, 2013)

    [Problema de minimización]
        min f(y) = (1/2) * y^T * Σ * y - Σ ln(y_k)
        
    [Restricciones]
        y_k > 0
        
    [Recuperación de los Pesos Finales (w)]
        w_k = y_k / Σ y_j
        
    """
    # 1. Semilla inicial: Volatility Timing (VT)
    y0 = W_VT(Sigma, K)
    
    # 2. Función objetivo: 0.5 * y^T * Sigma * y - sum(ln(y))
    def objective(y):
        y = np.maximum(y, 1e-10) 
        return 0.5 * y.T @ Sigma @ y - np.sum(np.log(y))
        
    # 3. Jacobiano (Gradiente exacto)
    # d f(y) / d y = Sigma * y - (1 / y)
    def jacobiano(y):
        y = np.maximum(y, 1e-10)
        return Sigma @ y - (1.0 / y)
        
    # 4. Restricciones
    bounds = tuple((1e-8, None) for _ in range(K))
    
    # 5. Optimización L-BFGS-B
    res = minimize(
        fun=objective,
        x0=y0,
        method='L-BFGS-B',
        jac=jacobiano,
        bounds=bounds,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    # 6. Resultado
    if res.success:
        y_star = res.x
        w_star = y_star / np.sum(y_star)
        return w_star
    else:
        return W_VT(Sigma, K)

def W_MVS(R_is, mu, Sigma, K, gamma=3.0):
    """
    CARTERA MVS (Mean-Variance-Skewness).
    Aproximación Estricta de Taylor para Utilidad

    [Problema de Maximización Original]
        max U(w) = -exp(-gamma * w^T * mu) * [1 + (gamma^2 / 2)*m2(w) - (gamma^3 / 6)*m3(w)]
        
    [Donde]
        m2(w) = w^T * Sigma * w
        m3(w) = w^T * M3 * (w ⊗ w)
        gamma  = Coeficiente único de aversión absoluta al riesgo
    
    Donde M3 es la matriz de co-asimetría empírica de dimensión (K x K^2).
    """
    R_is = np.asarray(R_is, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    if K == 1:
        return np.array([1.0])

    T_dias = float(R_is.shape[0])
    R_cen = R_is - mu  # Matriz de retornos centrados (T x K)
    M3 = np.zeros((K, K**2), dtype=float)
    
    for t in range(int(T_dias)):
        r_t = R_cen[t, :].reshape(K, 1)         # Vector columna (K, 1)
                                                # r_t.T es (1, K). Su Kronecker consigo mismo es (1, K^2)
        r_t_kron = np.kron(r_t.T, r_t.T)        # (1, K^2)
        M3 += r_t @ r_t_kron                    # (K, 1) @ (1, K^2) = (K, K^2)
        
    M3 = M3 / T_dias
    
    def objective(w):

        # Media muestral: mu_p(w) = w^T * mu
        mu_p = float(w.T @ mu)

        # Varianza muestral: m2(w) = w^T * Sigma * w
        m2 = float(w.T @ Sigma @ w)

        # Asimetría con Tensor: w^T * M3 * (w ⊗ w)
        w_kron_w = np.kron(w, w)
        m3 = float(w.T @ M3 @ w_kron_w)

        Term = 1.0 + (gamma**2 / 2.0)*m2 - (gamma**3 / 6.0)*m3
        return np.exp(-gamma * mu_p) * Term

    def jacobian(w):
        mu_p = float(w.T @ mu)
        m2 = float(w.T @ Sigma @ w)
        
        w_kron_w = np.kron(w, w)
        m3 = float(w.T @ M3 @ w_kron_w)

        Term = 1.0 + (gamma**2 / 2.0)*m2 - (gamma**3 / 6.0)*m3

        # Gradiente Varianza: 2 * Sigma * w
        grad_m2 = 2.0 * (Sigma @ w)
        
        # Gradiente Asimetría con Tensor: 3 * M3 * (w ⊗ w)
        grad_m3 = 3.0 * (M3 @ w_kron_w)

        grad_Term = (gamma**2 / 2.0) * grad_m2 - (gamma**3 / 6.0) * grad_m3
        grad_util = np.exp(-gamma * mu_p) * (-gamma * mu * Term + grad_Term)

        return grad_util

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = tuple((0.0, 1.0) for _ in range(K))
    x0 = W_EW(K)

    res = minimize(
        fun=objective,
        x0=x0,
        method='SLSQP',
        jac=jacobian,
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
    )

    if res.success:
        w_star = np.clip(res.x, 0.0, 1.0)
        w_sum = np.sum(w_star)
        if w_sum > 0.0:
            return w_star / w_sum

    return x0

def W_HRP(Sigma, K):
    """
    CARTERA HIERARCHICAL RISK PARITY (HRP).
    """
    
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = (Sigma + Sigma.T) / 2.0
    diag = np.diag(Sigma)

    if np.any(diag <= 1e-12):
        Sigma = Sigma.copy()
        np.fill_diagonal(Sigma, np.maximum(diag, 1e-12))

    w_hrp, _, _ = HRP(Sigma)
    w_hrp = np.clip(w_hrp, 0.0, 1.0)
    w_sum = w_hrp.sum()

    if w_sum > 0.0:
        return w_hrp / w_sum

    return W_EW(K)

def W_HERC(Sigma, K):
    """
    CARTERA HIERARCHICAL EQUAL RISK CONTRIBUTION (HERC) CLÁSICA.
    """

    Sigma = np.asarray(Sigma, dtype=float)
    # Forzamos simetría perfecta para evitar problemas de redondeo
    Sigma = (Sigma + Sigma.T) / 2.0

    # Estabilizamos numéricamente la diagonal para evitar divisiones por cero.
    diag = np.diag(Sigma)
    floor = 1e-12
    if np.any(diag <= floor):
        Sigma = Sigma.copy()
        np.fill_diagonal(Sigma, np.maximum(diag, floor))

    w_herc, _, _ = HERC(Sigma)
    
    # Aplanamos por seguridad y aplicamos clip
    w_herc = np.clip(np.asarray(w_herc).flatten(), 0.0, 1.0)
    w_sum = w_herc.sum()

    if w_sum > 0.0:
        return w_herc / w_sum

    return W_EW(K)
