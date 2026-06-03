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
    mu_plus = np.maximum(mu, 0.0)
    variances = np.diag(Sigma)
    variances = np.where(variances > 0, variances, np.inf)
    ratio = mu_plus / variances
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    sum_ratio = ratio.sum()
    
    if sum_ratio == 0 or np.isnan(sum_ratio):
        return W_EW(K)
    
    return ratio / sum_ratio


def W_GMV(R_IS: np.ndarray, K: int, print_delta: bool = False) -> np.ndarray:
    """
    CARTERA DE MÍNIMA VARIANZA GLOBAL (GMV).
    Busca la combinación de activos que minimiza la volatilidad total de la cartera, 
    ignorando las rentabilidades esperadas (Markowitz, 1952).
    Denoising mediante Shrinkage de Ledoit-Wolf para mejorar la estabilidad numérica de Sigma.

    [Problema de minimización]
        min f(w) = (1/2) * w^T * Σ * w
        
    [Restricciones]
        Σ w_k = 1      (Plena inversión)
        0 <= w_k <= 1  (Long-only)
        
    [Pesos Finales (w) normalizados]
        w_k = w*_k / Σ w*_j
    """
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
    w0 = W_EW(K)

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
    def objective(y):
        y = np.maximum(y, 1e-10) 
        return 0.5 * y.T @ Sigma @ y - np.sum(np.log(y))
        
    def jacobiano(y):
        y = np.maximum(y, 1e-10)
        return Sigma @ y - (1.0 / y)
        
    bounds = tuple((1e-8, None) for _ in range(K))
    y0 = W_VT(Sigma, K)

    res = minimize(
        fun=objective,
        x0=y0,
        method='L-BFGS-B',
        jac=jacobiano,
        bounds=bounds,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    if res.success:
        y_star = res.x
        w_star = y_star / np.sum(y_star)
        return w_star
    else:
        return W_VT(Sigma, K)

def W_MVS(R_is: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, K: int, gamma: float = 3.0) -> np.ndarray:
    """
    CARTERA DE MEDIA-VARIANZA-ASIMETRÍA (MVS).
    Maximiza la utilidad esperada de un inversor con aversión al riesgo constante (CARA),
    incorporando el tercer momento para preferir carteras con asimetría positiva 
    y penalizar el riesgo de cola severo. (Jondeau & Rockinger, 2006).

    [Problema de Minimización Equivalente] (Inversa de la Max. de Utilidad)
        min f(w) = exp(-gamma * w^T * mu) * [1 + (gamma^2 / 2)*m2(w) - (gamma^3 / 6)*m3(w)]
        
    [Jacobiano Analítico (Gradiente Exacto)]
        Grad_f(w) = exp(-gamma * mu_p) * [ -gamma * mu * Term + Grad_Term ]
        
        Donde los gradientes parciales de los momentos son:
        Grad_m2(w) = 2 * Sigma * w
        Grad_m3(w) = 3 * M3 * (w ⊗ w)
        Grad_Term  = (gamma^2 / 2) * Grad_m2(w) - (gamma^3 / 6) * Grad_m3(w)
        
    [Restricciones]
        Σ w_k = 1      (Plena inversión)
        0 <= w_k <= 1  (Solo posiciones largas)
    """
    R_is = np.asarray(R_is, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)

    T_dias = float(R_is.shape[0])
    R_cen = R_is - mu
    
    # Co-asimetría (M3) Vectorizada
    R_cen_exp1 = R_cen[:, :, np.newaxis]  # (T, K, 1)
    R_cen_exp2 = R_cen[:, np.newaxis, :]  # (T, 1, K)
    
    # Matriz plana de (T x K^2)
    R_cen_kron = (R_cen_exp1 * R_cen_exp2).reshape(int(T_dias), K**2)
    
    # Producto matricial final: (K x T) @ (T x K^2) -> (K x K^2)
    M3 = (R_cen.T @ R_cen_kron) / T_dias

    def objective(w):
        mu_p = np.dot(w, mu)
        m2 = np.dot(w, Sigma @ w)
        w_kron_w = np.kron(w, w)
        m3_val = np.dot(w, M3 @ w_kron_w)
        Term = 1.0 + (gamma**2 / 2.0)*m2 - (gamma**3 / 6.0)*m3_val
        return np.exp(-gamma * mu_p) * Term

    def jacobian(w):
        mu_p = np.dot(w, mu)
        m2 = np.dot(w, Sigma @ w)
        w_kron_w = np.kron(w, w)
        m3_val = np.dot(w, M3 @ w_kron_w)
        Term = 1.0 + (gamma**2 / 2.0)*m2 - (gamma**3 / 6.0)*m3_val
        grad_m2 = 2.0 * (Sigma @ w)
        grad_m3 = 3.0 * (M3 @ w_kron_w)
        grad_Term = (gamma**2 / 2.0) * grad_m2 - (gamma**3 / 6.0) * grad_m3
        grad_util = np.exp(-gamma * mu_p) * (-gamma * mu * Term + grad_Term)

        return grad_util

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = tuple((0.0, 1.0) for _ in range(K))
    x0 = np.full(K, 1.0 / K)

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

    Asigna el capital leyendo la estructura del mercado mediante Machine Learning 
    no supervisado. Evita la inversión de la matriz de covarianzas, erradicando la inestabilidad 
    numérica del modelo tradicional de Markowitz (López de Prado, 2016).

    [Algoritmo]
        1. Agrupación Jerárquica: Mapea la distancia entre activos usando 'Single Linkage'.
        2. Cuasi-Diagonalización: Reordena Σ agrupando activos correlacionados en la diagonal.
        3. Bisección Recursiva: Corta la lista a ciegas por la mitad y aplica Paridad de 
           Riesgo (Inversa de la Varianza) entre las ramas resultantes.
        
    [Restricciones]
        Σ w_k = 1      (Plena inversión)
        0 <= w_k <= 1  (Solo posiciones largas, garantizado topológicamente)
    """
    
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = (Sigma + Sigma.T) / 2.0
    diag = np.diag(Sigma)

    if np.any(diag <= 1e-12):
        Sigma = Sigma.copy()
        np.fill_diagonal(Sigma, np.maximum(diag, 1e-12))

    # Extraemos los pesos HRP
    w_hrp, _, _ = HRP(Sigma)

    # Restricciones de no negatividad y normalización
    w_hrp = np.clip(w_hrp, 0.0, 1.0)
    w_sum = w_hrp.sum()

    if w_sum > 0.0:
        return w_hrp / w_sum

    return W_EW(K)

def W_HERC(Sigma, K):
    """
    CARTERA HIERARCHICAL EQUAL RISK CONTRIBUTION (HERC).

    Combina Machine Learning (clustering jerárquico con método de Ward) y Paridad 
    de Riesgo para asignar el capital. (Raffinot, 2018).

    [Mecánica Algorítmica (Top-Down)]
        1. Agrupación Compacta (Ward): Link mediante método de Ward.
        2. Forzamos a N ≈ sqrt(K) clústeres para evitar el sobreajuste (overfitting).
        3. Asignación Inter-clúster (ERC)
        4. Asignación Intra-clúster (IVP)
        
    [Restricciones]
        Σ w_k = 1      (Plena inversión)
        0 <= w_k <= 1  (Long-only)
    """

    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = (Sigma + Sigma.T) / 2.0
    diag = np.diag(Sigma)
    floor = 1e-12
    if np.any(diag <= floor):
        Sigma = Sigma.copy()
        np.fill_diagonal(Sigma, np.maximum(diag, floor))

    # Extraemos los pesos HERC
    w_herc, _, _ = HERC(Sigma)
    
    # Restricciones de no negatividad y normalización
    w_herc = np.clip(np.asarray(w_herc).flatten(), 0.0, 1.0)
    w_sum = w_herc.sum()

    if w_sum > 0.0:
        return w_herc / w_sum

    return W_EW(K)