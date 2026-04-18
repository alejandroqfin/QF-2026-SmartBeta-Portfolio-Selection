"""
PERFORMANCE RATIOS: HERRAMIENTAS DE SCREENING Y PERFORMANCE
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martínez
"""

import numpy as np
import pandas as pd

def HISTORICAL_VAR(matrix: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """VaR histórico empírico: VaR_tau = Q_tau(R)."""
    if not (0 < tau < 1):
        raise ValueError("Tau debe estar estrictamente en (0, 1).")

    return np.quantile(matrix, tau, axis=0)

def SHARPE_RATIO(matrix: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Ratio de Sharpe: SR = (μ - rf) / σ."""
    mu = np.mean(matrix, axis=0)
    sigma = np.std(matrix, axis=0, ddof=1)
    return (mu - rf) * np.sqrt(252) / sigma

def MSR(matrix: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Marginal Sharpe Ratio (MSR) por activo respecto a cartera EW.
    MSR_i = (sigma_i / sigma_p) * SR_i - (Cov(r_i, r_p) / sigma_p^2) * SR_p"""
    T, K = matrix.shape
    
    # 1. Pesos de la cartera Base (EW)
    w_ew = np.ones(K) / K
    
    # 2. Momentos de la cartera base (EW)
    r_p = matrix @ w_ew
    mu_p = np.mean(r_p)
    sigma_p = np.std(r_p, ddof=1)
    sr_p = (mu_p - rf) / sigma_p
    
    # 3. Momentos individuales de los activos (i)
    mu_i = np.mean(matrix, axis=0)           # (K x 1)
    sigma_i = np.std(matrix, axis=0, ddof=1) # (K x 1)
    sr_i = (mu_i - rf) / sigma_i             # (K x 1)
    
    # 4. Covarianza con la cartera
    Sigma = np.cov(matrix, rowvar=False, ddof=1)
    cov_ip = Sigma @ w_ew                    # (K x 1)
    
    # 5. MSR
    msr = (sigma_i / sigma_p) * sr_i - (cov_ip / (sigma_p ** 2)) * sr_p
    
    return msr * np.sqrt(252)

def SHARPE_RATIO_CORR(window_matrix: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """
    Sharpe Ratio modificado penalizando por correlación.
    SR_i* = SR_i * (1 - media_i)
    """
    sr_classic = SHARPE_RATIO(window_matrix, rf=rf)

    # Matriz de correlación de Pearson
    corr_matrix = np.corrcoef(window_matrix, rowvar=False)
    abs_corr = np.abs(corr_matrix)

    # Media
    N = window_matrix.shape[1] 
    media_corr = (np.sum(abs_corr, axis=0) - 1.0) / (N - 1)

    return sr_classic * (1.0 - media_corr)

def VARR(matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """VaR Ratio empírico por activo.
    VaRR(α) = |VaR(1 - α)| / |VaR(α)|
    """
    var_lower = np.abs(HISTORICAL_VAR(matrix, tau=alpha))
    var_upper = np.abs(HISTORICAL_VAR(matrix, tau=1.0 - alpha))
    ratio = np.divide(
        var_upper,
        var_lower,
        out=np.full_like(var_upper, np.nan, dtype=float),
        where=var_lower != 0,
    )
    return ratio

def LPM(matrix: np.ndarray, theta: float = 0.0, m: float = 2.0) -> np.ndarray:
    """Lower Partial Moment empírico por activo.
    LPM(m, θ) = (1/T) * sum_t max(θ - R_t, 0)^m
    """
    matrix = np.asarray(matrix, dtype=float)
    return np.mean(np.maximum(theta - matrix, 0.0) ** m, axis=0)


def UPM(matrix: np.ndarray, theta: float = 0.0, q: float = 2.0) -> np.ndarray:
    """Upper Partial Moment empírico por activo.
    UPM(q, θ) = (1/T) * sum_t max(R_t - θ, 0)^q
    """
    matrix = np.asarray(matrix, dtype=float)
    return np.mean(np.maximum(matrix - theta, 0.0) ** q, axis=0)

def SORTINO_SATCHELL(matrix: np.ndarray, theta: float = 0.0, m: float = 2.0) -> np.ndarray:
    """Sortino-Satchell.
    SS = (μ - θ) / LPM^(1/m)
    """
    matrix = np.asarray(matrix, dtype=float)
    numerator = np.mean(matrix, axis=0) - theta
    denominator = LPM(matrix, theta=theta, m=m) ** (1.0 / m)
    
    return numerator / denominator * np.sqrt(252)


def FARINELLI_TIBILETTI(matrix: np.ndarray, theta: float = 0.0, q: float = 1.0, m: float = 2.0) -> np.ndarray:
    """Farinelli-Tibiletti ratio por activo.
    FT = UPM^(1/q) / LPM^(1/m)
    """
    matrix = np.asarray(matrix, dtype=float)
    
    numerator = UPM(matrix, theta=theta, q=q) ** (1.0 / q)
    denominator = LPM(matrix, theta=theta, m=m) ** (1.0 / m)
    
    return numerator / denominator

def GENERALIZED_RACHEV(
    matrix: np.ndarray,
    alpha: float = 0.05,
    theta: float = 0.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> np.ndarray:
    """Generalized Rachev Ratio (GR).

    Evalúa el exceso de retorno esperado en la cola derecha (potencia gamma) 
    frente a la pérdida esperada en la cola izquierda (potencia delta), 
    relativos a un umbral theta.

    Args:
        matrix: Array 2D (T x N) con los rendimientos.
        alpha: Nivel de significancia de las colas (ej. 0.05 para el 5%).
        theta: Umbral de retorno (0.0 por defecto).
        gamma: Exponente para la cola derecha (ganancias).
        delta: Exponente para la cola izquierda (pérdidas).
    """
    matrix = np.asarray(matrix, dtype=float)

    # Identificamos los cuantiles para aislar las colas
    q_left = np.percentile(matrix, alpha * 100.0, axis=0)
    q_right = np.percentile(matrix, (1.0 - alpha) * 100.0, axis=0)

    # 1 si pertenece a la cola, 0 si no
    mask_right = matrix >= q_right
    mask_left = matrix <= q_left

    # Operador de parte positiva para los excesos respecto a theta
    excess_right = np.maximum(matrix - theta, 0.0)
    excess_left = np.maximum(theta - matrix, 0.0)

    tail_right_values = (excess_right ** gamma) * mask_right
    tail_left_values = (excess_left ** delta) * mask_left

    right_mean = np.sum(tail_right_values, axis=0) / np.sum(mask_right, axis=0)
    left_mean = np.sum(tail_left_values, axis=0) / np.sum(mask_left, axis=0)

    numerator = right_mean ** (1.0 / gamma)
    denominator = left_mean ** (1.0 / delta)

    return numerator / denominator

def ROLLING_WINDOW_RANKINGS(
    df_returns: pd.DataFrame, 
    oos_start_date: str, 
    K: int, 
    ratios_list: list, 
    M: int = 1000
) -> tuple:
    """
   Motor de Screening y Rankings Out-Of-Sample.

   Estimación por ventana móvil (rolling-window) 
   
   Para cada instante t en el periodo Out-of-Sample (OOS), el algoritmo aísla una matriz 
   temporal estricta de M observaciones pasadas (In-Sample). Sobre este bloque matricial, 
   calcula de forma vectorizada las métricas y los momentos centrales 
   especificadas en `ratios_list`. Finalmente, ordena los N activos para seleccionar 
   el subconjunto de los K activos superiores.

   Args:
       df_returns (pd.DataFrame): Matriz de rendimientos empíricos de dimensión (T x N).
       oos_start_date (str): Fecha de inicio del periodo Out-Of-Sample. 
                             La primera iteración evalúa exclusivamente la información t < oos_start.
       K (int): Activos que conforman la cartera
       ratios_list (list[str]): Vector de métricas de performance a evaluar (ej. 'SR', 'VaRR5', 'GR5' ...)
       M (int): Tamaño de la ventana (ej. 1000 días de mercado).

   Returns:
       tuple:
           - dict_rankings (dict): DataFrames de dimensión (T x K) con los IDs 
                                   de los activos ganadores para cada día y cada ratio.
           - dict_series (dict): DataFrames de dimensión (T x N) con las series 
                                 temporales de la métrica para cada activo.
   """
    oos_start_date = pd.to_datetime(oos_start_date)
    
    # En qué fila exacta del histórico empieza nuestra simulación
    start_idx = int(df_returns.index.searchsorted(oos_start_date))
    
    # Las fechas de nuestros rankings (empezamos un día antes para invertir hoy)
    dates_rankings = df_returns.index[start_idx - 1 :]
    cols = df_returns.columns
    arr_returns = df_returns.values

    dict_rankings = {ratio: [] for ratio in ratios_list}
    dict_series = {ratio: [] for ratio in ratios_list}

    # El bucle empieza estrictamente donde inicia nuestro OOS
    for t in range(start_idx, len(df_returns) + 1):
        
        # Ventana constante de tamaño M
        window_matrix = arr_returns[t - M : t, :]

        for ratio in ratios_list:
            if ratio == 'SR':
                score = SHARPE_RATIO(window_matrix, rf=0.0)
            elif ratio == 'SR_Correlation':
                score = SHARPE_RATIO_CORR(window_matrix, rf=0.0)
            elif ratio == 'MSR':
                score = MSR(window_matrix, rf=0.0)
            elif ratio == 'VaRR1':
                score = VARR(window_matrix, alpha=0.01)
            elif ratio == 'VaRR5':
                score = VARR(window_matrix, alpha=0.05)
            elif ratio == 'VaRR10':
                score = VARR(window_matrix, alpha=0.10)
            elif ratio == 'Omega':
                score = FARINELLI_TIBILETTI(window_matrix, theta=0.0, q=1.0, m=1.0)
            elif ratio == 'UPR':
                score = FARINELLI_TIBILETTI(window_matrix, theta=0.0, q=1.0, m=2.0)
            elif ratio == 'Kappa3':
                score = SORTINO_SATCHELL(window_matrix, theta=0.0, m=3.0)
            elif ratio == 'Sortino':
                score = SORTINO_SATCHELL(window_matrix, theta=0.0, m=2.0)
            elif ratio == 'GR1':
                score = GENERALIZED_RACHEV(window_matrix, alpha=0.01, theta=0.0, gamma=1.0, delta=1.0)
            elif ratio == 'GR5':
                score = GENERALIZED_RACHEV(window_matrix, alpha=0.05, theta=0.0, gamma=1.0, delta=1.0)
            else:
                raise ValueError(f"Ratio no soportado en ROLLING_WINDOW_RANKINGS: {ratio}")

            dict_series[ratio].append(score)
            
            score_safe = np.where(np.isnan(score), -np.inf, score)
            topk_idx = np.argsort(score_safe)[-K:][::-1]
            dict_rankings[ratio].append(cols[topk_idx].tolist())

    cols_rank = [f'Top {j+1}' for j in range(K)]

    dict_rankings = {
        ratio: pd.DataFrame(values, index=dates_rankings, columns=cols_rank)
        for ratio, values in dict_rankings.items()
    }
    dict_series = {
        ratio: pd.DataFrame(values, index=dates_rankings, columns=cols)
        for ratio, values in dict_series.items()
    }

    return dict_rankings, dict_series

