"""
METRICS: HERRAMIENTAS MATEMÁTICAS
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martínez
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco
from sklearn.covariance import ledoit_wolf

def mean(df_log: pd.DataFrame) -> pd.Series:
    """ Esperanza: μ = (1 / T) * Σ r_t """
    T = df_log.shape[0]
    return df_log.sum(axis=0) / T

def volatility(df_log: pd.DataFrame) -> pd.Series:
    """ Desviación típica muestral: σ = sqrt( (1/(T-1)) * Σ (r_t - μ)^2 ) """
    T = df_log.shape[0]
    mu = mean(df_log)
    deviations = df_log - mu
    variance = (deviations ** 2).sum(axis=0) / (T - 1)
    return np.sqrt(variance)

def skewness(df_log: pd.DataFrame) -> pd.Series:
    """ Asimetría: S = m_3 / m_2^(3/2) """
    T = df_log.shape[0]
    mu = mean(df_log)
    deviations = df_log - mu
    m2 = (deviations ** 2).sum(axis=0) / T
    m3 = (deviations ** 3).sum(axis=0) / T
    return m3 / (m2 ** 1.5)

def kurtosis(df_log: pd.DataFrame) -> pd.Series:
    """Curtosis: K = (m_4 / m_2^2) 
    - Para una distribución normal, K = 3."""
    T = df_log.shape[0]
    mu = mean(df_log)
    deviations = df_log - mu
    m2 = (deviations ** 2).sum(axis=0) / T
    m4 = (deviations ** 4).sum(axis=0) / T
    return (m4 / (m2 ** 2))

def get_efficient_frontier(arithmetic_returns, num_portfolios=50, risk_free_rate=0.0, annual_factor=252):
    """Calcula la Frontera Eficiente optimizando las carteras (Vectorizado)."""
    mean_returns = arithmetic_returns.mean() * annual_factor
    cov_matrix = arithmetic_returns.cov() * annual_factor
    num_assets = len(mean_returns)

    # NUEVAS FUNCIONES ANIDADAS: Álgebra matricial directa (Vectorizada con @)
    def portfolio_return(weights):
        return float(weights.T @ mean_returns)

    def portfolio_volatility(weights):
        return float(np.sqrt(weights.T @ cov_matrix @ weights))

    def negative_sharpe(weights):
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        return -(ret - risk_free_rate) / vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Semilla inicial equiponderada limpia con numpy
    initial_guess = np.full(num_assets, 1.0 / num_assets)

    # A. Maximizar Ratio de Sharpe (MSRP)
    opt_sharpe = sco.minimize(negative_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # B. Minimizar Volatilidad (MVP)
    opt_vol = sco.minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # C. Calcular la Frontera Eficiente
    # Evaluamos retornos objetivo desde la cartera de mínima varianza hasta el activo con mayor retorno
    target_returns = np.linspace(
        portfolio_return(opt_vol.x), 
        mean_returns.max(),          
        num_portfolios
    )

    efficient_portfolios = []
    for target in target_returns:
        # Añadimos la restricción de que la cartera debe alcanzar el retorno objetivo
        eff_constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target}
        )
        eff_opt = sco.minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=eff_constraints)
        
        if eff_opt.success:
            efficient_portfolios.append({
                'Return': target,
                'Volatility': eff_opt.fun,
                'Weights': eff_opt.x
            })

    df_efficient_frontier = pd.DataFrame(efficient_portfolios)
    return df_efficient_frontier, opt_sharpe, opt_vol, mean_returns, cov_matrix

def replacement_rate(
    df_rankings: pd.DataFrame,
    fechas_oos: pd.DatetimeIndex,
    return_frequency: bool = False
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """
    Rebalanceo de cartera
    Calcula el número de nuevos ETFs que entran en la cartera cada día.

    Si return_frequency=True, también devuelve el contador de frecuencias
    de 0..K cambios por día.
    """
    # 1. Extraemos índices y matriz (T x K)
    t_indices = df_rankings.index.get_indexer(fechas_oos)
    matriz_activos = df_rankings.iloc[t_indices].values
    
    # 2. Convertimos matrices 2D a tensores 3D (T-1, K, 1) vs (T-1, 1, K)
    A_t = matriz_activos[1:, :, None]
    A_t_prev = matriz_activos[:-1, None, :]
    
    # 3. Matriz booleana (T-1, K): True si el ETF i de hoy estaba ayer en la cartera
    activos_mantenidos = (A_t == A_t_prev).any(axis=2)
    
    # 4. Cantidad de ETFs que sobreviven y los que entran nuevos
    cantidad_mantenidos = activos_mantenidos.sum(axis=1)
    K = matriz_activos.shape[1]
    nuevos_por_dia = K - cantidad_mantenidos

    rotacion_total = np.concatenate(([0], nuevos_por_dia))
    replacement_series = pd.Series(rotacion_total, index=fechas_oos)

    if not return_frequency:
        return replacement_series

    freq = replacement_series.value_counts().sort_index()
    freq = freq.reindex(range(0, K + 1), fill_value=0)
    freq.name = 'Frecuencia'
    freq.index.name = 'Nuevos ETFs'
    return replacement_series, freq.astype(int)

def etf_stats(target_ids_matrix: np.ndarray, tickers_list: list) -> pd.DataFrame:
    """
    Calcula la tasa de supervivencia (frecuencia)
    y la posición mediana para el universo de N ETFs.
    """
    T, K = target_ids_matrix.shape
    N = len(tickers_list)

    days_active = np.bincount(target_ids_matrix.flatten(), minlength=N)
    survival_rate = days_active / T

    median_rank = np.full(N, np.nan)
    for i in range(N):
        if days_active[i] > 0:
            # np.where devuelve (rows, cols). Extraemos las columnas (posiciones 0 a K-1)
            # Sumamos 1 para que el rango sea natural (de 1 a K)
            posiciones = np.where(target_ids_matrix == i)[1] + 1
            median_rank[i] = np.median(posiciones)

    position_counts = np.zeros((N, K), dtype=int)
    for pos in range(K):
        position_counts[:, pos] = np.bincount(target_ids_matrix[:, pos], minlength=N)

    df_survival = pd.DataFrame({
        'Ticker': tickers_list,
        'Days_Active': days_active,
        'Survival_Rate': survival_rate,
        'Median_Position': median_rank
    })

    df_positions = pd.DataFrame(position_counts, columns=[f'Top_{i+1}' for i in range(K)])
    df_final = pd.concat([df_survival, df_positions], axis=1)

    df_final = df_final[df_final['Days_Active'] > 0].sort_values(by='Survival_Rate', ascending=False)

    return df_final.reset_index(drop=True)

def transaction_costs(returns_matrix, target_ids_matrix, c=0.0010):
    """
    Solo sirve para pesos equiponderados (EW) y para fricciones proporcionales (c).
    
    Args:
    returns_matrix: np.array (T, N) - Matriz estocástica de rendimientos OOS sobre el universo completo
    target_ids_matrix: np.array (T, K) - Índices (0 a N-1) de los K ETFs elegidos cada día
    c: float - Costes de transacción (ej. 0.0010 = 10 bps)
    """
    T, N = returns_matrix.shape
    K = target_ids_matrix.shape[1]
    
    # 1. Construcción de la matriz de pesos objetivo (T, N)
    # Para cada día t, asignamos pesos equiponderados (1/K) a los ETFs seleccionados y 0 al resto
    W_target = np.zeros((T, N))
    row_idx = np.arange(T)[:, None]          # Vector fila (T, 1) con índices de días
    W_target[row_idx, target_ids_matrix] = 1.0 / K
        
    # 2. Inicialización del vector de riqueza y estado inicial
    W_net = np.zeros(T)
    W_net[0] = 1.0  # Riqueza inicial = 1€
    
    # Pesos iniciales: asumimos que en t=0 se compra la cartera objetivo sin fricción (w_0 = w_target[0])
    w_prev = W_target[0, :].copy()
    
    # 3. Bucle de rebalanceo para cada día t = 1, 2, ..., T-1 (Eq. 16)
    for t in range(1, T):
        
        # A) Rendimiento Bruto del portafolio hoy (R_p = w' * r_t)
        r_p = w_prev @ returns_matrix[t, :]
        
        # B) Pesos después del movimiento natural del mercado (drifted weights)
        # w_drifted_i = w_prev_i * (1 + r_i,t) / (1 + r_p,t)
        w_drifted = w_prev * (1 + returns_matrix[t, :]) / (1 + r_p)
        
        # C) El peso objetivo dictado por el ranking para MAÑANA
        w_target = W_target[t, :]
        
        # D) Rebalanceo de ETFs: Sumamos cuánto se compra/vende de cada uno de los N ETFs
        turnover = np.sum(np.abs(w_target - w_drifted))
        
        # E) Eq. 16 DeMiguel
        W_net[t] = W_net[t-1] * (1 + r_p) * (1 - c * turnover)
        
        # F) Actualizamos el vector de pesos para la siguiente iteración
        w_prev = w_target.copy()
        
    return W_net

def risk_contribution(w: np.ndarray, cov_matrix: np.ndarray) -> tuple:
    """
    Descomposición de Euler del riesgo de la cartera.
    Retorna la Contribución al Riesgo Absoluta (RC) y Relativa (RC_rel).
    """
    # 1. Varianza y Volatilidad total de la cartera
    var_p = w.T @ cov_matrix @ w
    vol_p = np.sqrt(var_p)

    # Carteras con volatilidad nula
    if vol_p <= 1e-12:
        zeros = np.zeros_like(w, dtype=float)
        return zeros, zeros
    
    # 2. Vector de Riesgo Marginal (MR): d(sigma) / d(w) = (Sigma * w) / vol_p
    MR = (cov_matrix @ w) / vol_p
    
    # 3. Contribución Absoluta (RC) = w_i * MR_i
    RC = w * MR
    
    # 4. Contribución Relativa (RC_rel) = RC_i / vol_p
    # Forzamos a que la suma del RC_rel sea estrictamente 1.0 (100%)
    RC_rel = RC / vol_p
    return RC, RC_rel


def MHI(w: np.ndarray) -> float:
    """
    Modified Herfindahl Index (MHI).

    MHI = (sum(w^2) - 1/K) / (1 - 1/K)

    Para K = 1 devuelve 1.0.
    """
    w = np.asarray(w, dtype=float)
    K = w.size

    if K == 1:
        return 1.0

    return float(((w @ w) - (1.0 / K)) / (1.0 - (1.0 / K)))


def allocation_model_with_costs(
    T: pd.DatetimeIndex,
    t_indices: np.ndarray,
    M: int,
    K: int,
    rendimientos: np.ndarray,
    ids_matrix: np.ndarray,
    model: callable,
    perfil: str,
    c: float = 0.0020,
    save_mhi: callable = None,
    **model_kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulacion Out of Sample con costes de transacción.

    La riqueza diaria no solo absorbe la rentabilidad bruta de los activos elegidos, sino 
    que es penalizada por el volumen exacto de capital que el algoritmo se ve obligado 
    a mover (turnover) para alinear la cartera deformada por el mercado con la nueva cartera objetivo.

    [1. Ecuación de Riqueza Neta (DeMiguel et al., 2009)]
    W_t = W_{t-1} * (1 + R_{p,t}) * [1 - c * sum( |w_{i,t} - w^+_{i,t-1}| )]
    
    Donde:
    - W_t      : Riqueza neta patrimonial en el instante t.
    - R_{p,t}  : Rentabilidad bruta del portafolio en el día t.
    - c        : Coste de transacción proporcional (fricción escalar).
    - w_{i,t}  : Peso objetivo asignado al activo i por el motor de optimización hoy.
    - w^+_{i,t-1}: Peso real (desplazado) del activo i al cierre del mercado, heredado de ayer.

    [2. Ecuación de Deriva de Pesos (w+)]
    Para cada activo i individual en el universo N:
    w^+_{i,t-1} = w_{i,t-1} * [ (1 + r_{i,t}) / (1 + R_{p,t}) ]
    
    (Si un ETF sube un 5% hoy y el resto de la cartera no, su peso relativo dentro del 
    portafolio crece orgánicamente. El turnover se cobra estrictamente sobre la diferencia 
    entre este peso inflado y el peso que el algoritmo dicta para mañana).
    """
    
    # Numero total de ETFs (N = 100)
    N = rendimientos.shape[1]

    rendimientos_oos = np.zeros(len(T))  # (T x 1)
    riqueza_oos = np.zeros(len(T))       # (T x 1)

    w_plus = None       # Pesos al cierre del día anterior (w+)
    riqueza_prev = 1.0  # Empezamos con 1€

    # Bucle para cada día out of sample
    for i in range(len(T)):
        t_i = t_indices[i]              # escalar
        ids_t = ids_matrix[i]           # (K x 1)

        # FASE 1: IN SAMPLE (M dias previos a t_i)
        # Rendimientos de los K ETFs seleccionados (M x K)
        rendimientos_is = rendimientos[t_i - M: t_i, ids_t]

        # Rendimientos del universo completo  (N x 1)
        rendimientos_universo_t = rendimientos[t_i, :]

        # Media (K x 1) y Matriz de covarianzas (K x K) para los K ETFs seleccionados
        mu = rendimientos_is.mean(axis=0)                    
        Sigma = np.cov(rendimientos_is, rowvar=False, ddof=1)
        Sigma = (Sigma + Sigma.T) / 2.0

        # FASE 2: CUANTO CAPITAL ASIGNAR A CADA ETF (K x 1)
        if perfil == 'EW':
            w_k = model(K)
        elif perfil == 'VT':
            w_k = model(Sigma, K)
        elif perfil == 'RRT':
            w_k = model(mu, Sigma, K)
        elif perfil == 'MVS':
            w_k = model(rendimientos_is, mu, Sigma, K, **model_kwargs)
        elif perfil in ['GMV', 'ERC', 'HRP', 'HERC']:
            w_k = model(Sigma, K)
    
        # Proyectamos sobre N = 100 activos
        # Asignamos 0 a los no seleccionados
        w = np.zeros(N)

        # Asignamos w_k a los seleccionados
        w[ids_t] = w_k

        # Guardamos el MHI
        if save_mhi is not None:
            save_mhi(i, w_k)

        # FASE 3: COSTES DE TRANSACCION
        # 4 CASOS POSIBLES DE |w - w_plus|:

        #   1. Ayer SÍ, Hoy SÍ  : (Ej: |10% - 12%| = 2% rotado).
        #   2. Ayer SÍ, Hoy NO  : (Ej: |0% - 10%| = 10% rotado).
        #   3. Ayer NO, Hoy SÍ  : (Ej: |10% - 0%| = 10% rotado).
        #   4. Ayer NO, Hoy NO  : (Ej: |0% - 0%| = 0% rotado).

        if w_plus is None:
            turnover = 0.0   # Coste 0 para el caso t = 0
        else:
            turnover = float(np.sum(np.abs(w - w_plus)))

        # Retorno bruto de la cartera el día t
        rendimiento_bruto_t = float(w @ rendimientos_universo_t)

        # W_t = W_{t-1} * (1 + R_p) * (1 - c * Turnover)
        multiplicador_bruto = 1.0 + rendimiento_bruto_t
        multiplicador_neto = multiplicador_bruto * (1.0 - c * turnover)
        
        # Actualizamos riqueza neta
        riqueza_t = riqueza_prev * multiplicador_neto

        # Guardamos resultados OOS
        riqueza_oos[i] = riqueza_t
        rendimientos_oos[i] = (riqueza_t / riqueza_prev) - 1.0

        # FASE 4: CÁLCULO DE LOS PESOS AL CIERRE DEL DÍA (w+)
        if np.isclose(multiplicador_bruto, 0.0):
            w_plus = w.copy()
        else:
            w_plus = w * (1.0 + rendimientos_universo_t) / multiplicador_bruto

        # Actualizamos riqueza_prev para la siguiente iteración
        riqueza_prev = riqueza_t

    return rendimientos_oos, riqueza_oos


def shrinkage(R_IS: np.ndarray, print_delta: bool = False) -> np.ndarray:
    """
    Calcula la matriz de covarianzas encogida (Shrunk Covariance)
    usando el estimador analítico óptimo de Ledoit-Wolf (2004).
    """
    Sigma_shrinkage, delta_optimo = ledoit_wolf(R_IS)
    return Sigma_shrinkage, delta_optimo

def maxDrawDown(wealth_series: pd.Series) -> float:
    """
    Maximum Drawdown (MDD)
    Mide la mayor caída sufrida por una inversión entre dos marcas de agua altas (HWM).
    
    Lógica Vectorizada:
    1. Calcula el High-Water Mark (HWM) histórico instantáneamente con .cummax()
    2. Calcula el drawdown en cada punto t como (P_t - HWM_t) / HWM_t
    3. Retorna el peor escenario (el mínimo global).
    """
    # 1. HWM histórico en cada instante t
    hwm = wealth_series.cummax()
    
    # 2. Caída porcentual desde el HWM
    drawdowns = (wealth_series - hwm) / hwm
    
    # 3. Peor caída registrada
    return float(drawdowns.min())

def coincident_assets_median(ids_A, ids_B, K):
    """
    Coincidencia entre universos de inversión.
    Calcula cuántos activos comparten dos carteras en cada instante t y extrae la mediana.
    Sin tener en cuenta la posición en el ranking.
    """
    # Broadcasting: (T x K x 1) == (T x 1 x K) -> (T x K x K)
    match_matrix = (ids_A[:, :, np.newaxis] == ids_B[:, np.newaxis, :])
    
    # Sumamos coincidencias por día y calculamos el % sobre K
    coincidencia_t = np.sum(match_matrix, axis=(1, 2)) / K
    return np.median(coincidencia_t) * 100

