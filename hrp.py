"""
CARTERAS MACHINE LEARNING: HRP & HERC
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martínez
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

def correlDist(corr: pd.DataFrame) -> np.ndarray:
    """Transforma la matriz de correlaciones en un espacio métrico de distancias."""
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0)).values

def getQuasiDiag(link: np.ndarray) -> list:
    """Reordena la matriz basándose en el dendrograma.
    Coge el esquema del dendrograma y empieza a ordenar físicamente los activos en sus familias.
    Reemplaza cada cluster por sus subramas y va bajando recursivamente hasta llegar a las hojas.
    El resultado en python es una lista de indicies ordenadas que me indica en que posicion debe ir cada activo.
    Los riesgos más interconectados se agrupan a lo largo de la DP de la matriz y lo demás son valores cercanos a 0"""
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
    """Calcula el Inverse-Variance Portfolio (IVP)."""
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov: pd.DataFrame, cItems: list) -> float:
    """Calcula la varianza representativa de un cluster."""
    cov_ = cov.iloc[cItems, cItems]
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)
    return float(np.asarray(cVar).reshape(-1)[0])

def getRecBipart(cov: pd.DataFrame, sortIx: list) -> pd.Series:
    """Reparte el capital 100% de arriba hacia abajo (bisección recursiva)."""
    
    # Asignamos un peso provisional de 1 a cada activo
    w = pd.Series(1.0, index=sortIx)
    cItems = [sortIx]

    # Bucle que corta la lista por la mitad en cada paso (Bisección)
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i]
            cItems1 = cItems[i+1]
            
            # Para cada subgrupo calcula la varianza global
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            
            # Factor de escalado que equilibra los riesgos
            # Al más volatil le da menos capital
            denom = cVar0 + cVar1
            alpha = 0.5 if denom <= 1e-16 else 1.0 - cVar0 / denom
            
            # Actualiza los pesos
            w.iloc[cItems0] *= alpha
            w.iloc[cItems1] *= (1.0 - alpha)
    return w


def getCorrMatrix(cov: pd.DataFrame) -> pd.DataFrame:
    """Convierte una matriz de covarianzas en correlaciones de forma robusta."""
    cov = cov.copy()
    std = np.sqrt(np.clip(np.diag(cov.values), 1e-16, None))
    denom = np.outer(std, std)
    corr_values = cov.values / denom
    corr_values = np.clip(corr_values, -1.0, 1.0)
    np.fill_diagonal(corr_values, 1.0)
    return pd.DataFrame(corr_values, index=cov.index, columns=cov.columns)

# CARTERA HRP (LÓPEZ DE PRADO, 2016)
def HRP(Sigma: np.ndarray, labels: list = None, method: str = 'single') -> tuple:
    """
    Algoritmo HRP (Hierarchical Risk Parity) - Marcos López de Prado (2016).
    Sigma es una matriz de KxK componentes (numpy array).
    """
    K = Sigma.shape[0]

    if labels is None:
        labels = list(range(K))

    Sigma_df = pd.DataFrame(Sigma, index=labels, columns=labels)
    corr_df = np.clip(getCorrMatrix(Sigma_df), -1.0, 1.0)
    
    # Topología y Dendrograma (Single Linkage)
    dist = correlDist(corr_df)
    link = sch.linkage(ssd.squareform(dist, checks=False), method=method)
    
    # Cuasi-diagonalización
    sort_ix = getQuasiDiag(link)

    # Bisección Recursiva
    w_series = getRecBipart(Sigma_df, sort_ix)
    w_series = w_series.reindex(labels, fill_value=0.0)

    # Normalización
    w_sum = w_series.values.sum()
    if w_sum > 0.0 and not np.isnan(w_sum):
        W_HRP = w_series.values / w_sum
    else:
        W_HRP = np.zeros(K)
        
    Sigma_quasidiag = Sigma_df.iloc[sort_ix, sort_ix]
    
    return W_HRP, link, Sigma_quasidiag

def _get_leaves(link: np.ndarray, node: int, N: int) -> list:
    """
    Extraer las hojas originales (activos) 
    que cuelgan de un nodo específico del dendrograma.
    """
    if node < N:
        return [int(node)]
    else:
        left_child = int(link[node - N, 0])
        right_child = int(link[node - N, 1])
        return _get_leaves(link, left_child, N) + _get_leaves(link, right_child, N)

# CARTERA HERC (RAFFINOT, 2018)
def HERC(Sigma: np.ndarray, labels: list = None) -> tuple:
    """
    Algoritmo HERC (Hierarchical Equal Risk Contribution) - Thomas Raffinot (2018).
    Calcula internamente el número óptimo de clústeres usando la raíz cuadrada de N.
    """
    K = Sigma.shape[0]
    
    # Número de clústeres
    N_clusters = max(1, int(np.sqrt(K)))

    if labels is None:
        labels = list(range(K))

    Sigma_df = pd.DataFrame(Sigma, index=labels, columns=labels)

    # Topología y Dendrograma (Ward)
    dist = correlDist(getCorrMatrix(Sigma_df))
    link = sch.linkage(ssd.squareform(dist, checks=False), method='ward')
    sort_ix = getQuasiDiag(link)
    
    # Bisección Top-Down
    root_node = 2 * K - 2
    active_nodes = [root_node]
    node_weight = {root_node: 1.0}

    # BUCLE DE CORTAFUEGOS: Cortamos el árbol basándonos en N_clusters
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

    # Asignación Intra-clúster (IVP)
    w_series = pd.Series(0.0, index=labels)
    for node in active_nodes:
        leaves = _get_leaves(link, node, K)
        if len(leaves) == 1:
            w_series.iloc[leaves] = node_weight[node]
        else:
            w_series.iloc[leaves] = node_weight[node] * np.asarray(getIVP(Sigma_df.iloc[leaves, leaves])).flatten()

    # Normalización final y salidas
    W_HERC = w_series.values / w_series.values.sum()
    Sigma_quasidiag = Sigma_df.iloc[sort_ix, sort_ix]
    
    return W_HERC, link, Sigma_quasidiag
