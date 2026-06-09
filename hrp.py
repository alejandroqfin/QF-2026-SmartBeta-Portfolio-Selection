"""
MODELOS DE CLÚSTERING JERÁRQUICO MACHINE LEARNING: HRP & HERC
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martínez

Nota:
Este módulo implementa algoritmos de Machine Learning no supervisado para la 
asignación de activos basada en grafos y clústering jerárquico, siguiendo 
las formulaciones originales de M. López de Prado (2016) y T. Raffinot (2018).
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

def correlDist(corr: pd.DataFrame) -> np.ndarray:
    """Transforma la matriz de correlaciones en distancias físicas.
    Fórmula:
        d_ij = sqrt(0.5 * (1 - rho_ij))"""
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0)).values

def getQuasiDiag(link: np.ndarray) -> list:
    """
    Genera la lista de índices óptima leyendo la jerarquía del dendrograma desde la raíz hasta las hojas.
    Estas posiciones se utilizarán después para cuasi-diagonalizar la matriz de covarianzas
    y agrupar físicamente los activos más interconectados.
    """
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]

    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])

    return sortIx.tolist()

def getIVP(cov: pd.DataFrame) -> np.ndarray:
    """
    Calcula los pesos mediante la Inverse-Variance Portfolio (IVP).
    Asigna el capital de forma inversamente proporcional a la varianza de cada activo    
    
    Fórmula:
        w_i = (1 / sigma_i^2) / sum(1 / sigma_j^2)
    """
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov: pd.DataFrame, cItems: list) -> float:
    """Calcula la varianza representativa de un cluster.
    
    Fórmula:
        cVar = w^T * cov * w
    """
    cov_ = cov.iloc[cItems, cItems]
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)
    return float(np.asarray(cVar).reshape(-1)[0])

def getRecBipart(cov: pd.DataFrame, sortIx: list) -> pd.Series:
    """
    Asignación de capital (top-down) mediante bisección recursiva sobre el orden cuasi-diagonalizado.
    Divide iterativamente los clústeres en mitades y distribuye el capital de forma inversamente proporcional a la varianza de cada rama.
    
    Fórmulas de asignación del capital:
        alpha = 1 - (V_left / (V_left + V_right))
        w_left = w_left * alpha
        w_right = w_right * (1 - alpha)
    """
    
    w = pd.Series(1.0, index=sortIx)
    cItems = [sortIx]

    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i]
            cItems1 = cItems[i+1]
            
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            
            denom = cVar0 + cVar1
            alpha = 0.5 if denom <= 1e-16 else 1.0 - cVar0 / denom
            
            w.loc[cItems0] *= alpha
            w.loc[cItems1] *= (1.0 - alpha)
    return w


def getCorrMatrix(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte una matriz de covarianzas en una matriz de correlaciones.
    
    Fórmula:
        rho_ij = Sigma_ij / (sigma_i * sigma_j)
    """
    cov = cov.copy()
    std = np.sqrt(np.clip(np.diag(cov.values), 1e-16, None))
    denom = np.outer(std, std)
    corr_values = cov.values / denom
    corr_values = np.clip(corr_values, -1.0, 1.0)
    np.fill_diagonal(corr_values, 1.0)
    return pd.DataFrame(corr_values, index=cov.index, columns=cov.columns)

def HRP(Sigma: np.ndarray, labels: list = None) -> tuple:
    """
    Algoritmo Hierarchical Risk Parity (HRP) de Marcos López de Prado (2016).
    
    Utiliza clustering jerárquico (Machine Learning no supervisado) para asignar capital basándose 
    en la estructura de la matriz de covarianzas, mitigando la inestabilidad numérica de la 
    optimización tradicional de Markowitz. Emplea el método de aglomeración 'single' (linkage) 
    para el dendrograma, definiendo la distancia entre clústeres como la mínima entre sus elementos 
    más cercanos.

    Parameters
    ----------
    Sigma : np.ndarray
        Matriz de covarianzas de los activos (K x K).
    labels : list, optional
        Tickers o ID de los activos. Si es None, se asignan índices enteros (0 a K-1).

    Returns
    -------
    tuple
        - W_HRP (np.ndarray): Vector de pesos óptimos normalizados en el orden de entrada original.
        - link (np.ndarray): Matriz topológica (linkage matrix) generada por el dendrograma ('single').
        - Sigma_quasidiag (pd.DataFrame): Matriz de covarianzas reordenada físicamente (cuasi-diagonalizada).
    """

    # 1. Número de activos en la cartera
    K = Sigma.shape[0]

    if labels is None:
        labels = list(range(K))

    # Matriz de correlaciones a partir de la matriz de covarianzas
    Sigma_df = pd.DataFrame(Sigma, index=labels, columns=labels)
    corr_df = np.clip(getCorrMatrix(Sigma_df), -1.0, 1.0)
    
    # 2. Distancias y Dendrograma (Clustering Jerárquico 'single')
    dist = correlDist(corr_df)
    link = sch.linkage(ssd.squareform(dist, checks=False), method='single')
    
    # 3. Cuasi-diagonalización
    sort_ix = getQuasiDiag(link)

    # 4. Bisección Recursiva para el reparto del capital
    w_series = getRecBipart(Sigma_df, sort_ix)
    w_series = w_series.sort_index()

    # 5. Normalización de pesos
    w_sum = w_series.values.sum()
    if w_sum > 0.0 and not np.isnan(w_sum):
        W_HRP = w_series.values / w_sum
    else:
        W_HRP = np.zeros(K)
        
    Sigma_quasidiag = Sigma_df.iloc[sort_ix, sort_ix]
    
    return W_HRP, link, Sigma_quasidiag

def _get_leaves(link: np.ndarray, node: int, N: int) -> list:
    """
    Extrae de forma recursiva las hojas originales (ETFs) 
    que cuelgan de un nodo específico del dendrograma.

    Recorre de arriba hacia abajo para identificar los activos que pertenecen 
    a cada clúster, y posteriormente asignarles el capital.
    """
    if node < N:
        return [int(node)]
    else:
        left_child = int(link[node - N, 0])
        right_child = int(link[node - N, 1])
        return _get_leaves(link, left_child, N) + _get_leaves(link, right_child, N)

def HERC(Sigma: np.ndarray, labels: list = None) -> tuple:
    """
    Algoritmo Hierarchical Equal Risk Contribution (HERC) de Thomas Raffinot (2018).
    
    A diferencia de HRP, HERC utiliza el método de 'Ward' para formar las familias 
    y divide la asignación de capital en dos fases: 
            
            i) asignación inter-clúster (Top-Down usando ERC) 
            ii) asignación intra-clúster (Bottom-Up usando IVP). 

    Parameters
    ----------
    Sigma : np.ndarray
        Matriz de covarianzas de los activos (K x K).
    labels : list, optional
        Tickers o ID de los activos. Si es None, se asignan índices enteros (0 a K-1).

    Returns
    -------
    tuple
        - W_HERC (np.ndarray): Vector de pesos óptimos normalizados.
        - link (np.ndarray): Matriz topológica (linkage matrix) de Ward.
        - Sigma_quasidiag (pd.DataFrame): Matriz de covarianzas reordenada físicamente.
    """

    # 1. Número de activos en la cartera
    K = Sigma.shape[0]
    
    # 2. Número de clústeres
    N_clusters = max(1, int(np.sqrt(K)))

    if labels is None:
        labels = list(range(K))

    Sigma_df = pd.DataFrame(Sigma, index=labels, columns=labels)

    # 3. Topología y Dendrograma (Ward)
    dist = correlDist(getCorrMatrix(Sigma_df))
    link = sch.linkage(ssd.squareform(dist, checks=False), method='ward')
    sort_ix = getQuasiDiag(link)
    
    # 4. Bisección Top-Down
    root_node = 2 * K - 2
    active_nodes = [root_node]
    node_weight = {root_node: 1.0}

    # 5. Cortamos el dendrograma hasta obtener el número deseado de clústeres
    for _ in range(N_clusters - 1):
        split_node = max(active_nodes)
        active_nodes.remove(split_node)

        left = int(link[split_node - K, 0])
        right = int(link[split_node - K, 1])
        
        leaves_left = _get_leaves(link, left, K)
        leaves_right = _get_leaves(link, right, K)

        # Asignación Inter-clúster (ERC)
        var_left = getClusterVar(Sigma_df, leaves_left)
        alpha = 1.0 - (var_left / (var_left + getClusterVar(Sigma_df, leaves_right)))

        node_weight[left] = node_weight[split_node] * alpha
        node_weight[right] = node_weight[split_node] * (1.0 - alpha)
        active_nodes.extend([left, right])

    # 6. Asignación Intra-clúster (IVP)
    w_series = pd.Series(0.0, index=labels)
    for node in active_nodes:
        leaves = _get_leaves(link, node, K)
        if len(leaves) == 1:
            w_series.iloc[leaves] = node_weight[node]
        else:
            w_series.iloc[leaves] = node_weight[node] * np.asarray(getIVP(Sigma_df.iloc[leaves, leaves])).flatten()

    # 7. Normalización de pesos
    W_HERC = w_series.values / w_series.values.sum()
    Sigma_quasidiag = Sigma_df.iloc[sort_ix, sort_ix]
    
    return W_HERC, link, Sigma_quasidiag