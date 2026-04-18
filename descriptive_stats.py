"""
ANÁLISIS DESCRIPTIVO: ARITHMETIC, LOG RETURNS AND RISK METRICS
Smart Beta ETF Universe - TFM
Autor: Alejandro Martínez
"""

import pandas as pd
import numpy as np
import shutil
from metrics import (
    mean, 
    volatility, 
    skewness, 
    kurtosis,
    get_efficient_frontier
)
from screening import SHARPE_RATIO, HISTORICAL_VAR
from plots import plot_boxplots, plot_efficient_frontier

# 1. CARGAMOS EXCEL
FILE_PATH_IN = "100_ETFs_Smart_Beta.xlsx"
FILE_PATH_OUT = "100_ETFs_AnalisisDescriptivo.xlsx"
DATA_SHEET = "Precios_Historicos"

shutil.copy(FILE_PATH_IN, FILE_PATH_OUT)

# 2. MATRIZ DE PRECIOS
price_matrix = pd.read_excel(
    FILE_PATH_IN, 
    sheet_name=DATA_SHEET, 
    index_col=0, 
    header=[0, 1], 
    parse_dates=True
)

T, N = price_matrix.shape
print(f"Observaciones (T): {T}")
print(f"ETFs (N): {N}")


# 3. CÁLCULO DE RENDIMIENTOS
log_returns = np.log(price_matrix / price_matrix.shift(1)).dropna(axis=0, how='all')
arithmetic_returns = price_matrix.pct_change().dropna(axis=0, how='all')

# 4. CÁLCULO DE MÉTRICAS (Matriz N x M unificada)
annual_factor = 252

df_metrics = pd.DataFrame({
    'Mean': mean(log_returns) * annual_factor,
    'Volatility': volatility(log_returns) * np.sqrt(annual_factor),
    'Skewness': skewness(log_returns),
    'Kurtosis': kurtosis(log_returns),
    'Sharpe_Ratio': SHARPE_RATIO(log_returns, rf=0.0) * np.sqrt(annual_factor),
    'VaR_01': HISTORICAL_VAR(log_returns, tau=0.01),
    'VaR_05': HISTORICAL_VAR(log_returns, tau=0.05),
    'VaR_10': HISTORICAL_VAR(log_returns, tau=0.10),
    'VaR_90': HISTORICAL_VAR(log_returns, tau=0.90),
    'VaR_95': HISTORICAL_VAR(log_returns, tau=0.95), 
    'VaR_99': HISTORICAL_VAR(log_returns, tau=0.99) 
})

# VAR RATIOS
df_metrics['VaRR_1'] = np.abs(df_metrics['VaR_01']) / np.abs(df_metrics['VaR_99']).replace(0, np.nan)
df_metrics['VaRR_5'] = np.abs(df_metrics['VaR_05']) / np.abs(df_metrics['VaR_95']).replace(0, np.nan)
df_metrics['VaRR_10'] = np.abs(df_metrics['VaR_10']) / np.abs(df_metrics['VaR_90']).replace(0, np.nan)

# RANKINGS DE MEJOR A PEOR
df_metrics['Ranking_Sharpe'] = df_metrics['Sharpe_Ratio'].rank(ascending=False)
df_metrics['Ranking_VaRR_5'] = df_metrics['VaRR_5'].rank(ascending=False)
df_metrics['Ranking_VaRR_1'] = df_metrics['VaRR_1'].rank(ascending=False)
df_metrics['Ranking_VaRR_10'] = df_metrics['VaRR_10'].rank(ascending=False)

# 5. CORRELACIÓN DE SPEARMAN
rank_columns = ['Ranking_Sharpe', 'Ranking_VaRR_1', 'Ranking_VaRR_5', 'Ranking_VaRR_10']
spearman_corr_matrix = df_metrics[rank_columns].corr(method='spearman')

print("\nMatriz de Correlación de Spearman")
print(spearman_corr_matrix)


# 6. EXPORTAMOS RESULTADOS AL EXCEL
with pd.ExcelWriter(FILE_PATH_OUT, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    log_returns.to_excel(writer, sheet_name='Log Returns')
    arithmetic_returns.to_excel(writer, sheet_name='Arithmetic Returns')
    df_metrics.to_excel(writer, sheet_name='Risk Metrics Summary')
    spearman_corr_matrix.to_excel(writer, sheet_name='Spearman Correlation')

print(f"\nMétricas calculadas en el archivo: {FILE_PATH_OUT}.")

# 7. BOX-PLOTS
plot_boxplots(df_metrics)

# 8. TABLA DE ESTADISTICOS PRINCIPALES
cols_objetivo = ['Mean', 'Volatility', 'Skewness', 'Kurtosis', 'VaR_01', 'VaR_05', 'VaR_10']
df_cross = df_metrics[cols_objetivo].copy()

quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
df_mean = df_cross.mean().to_frame(name='Mean')

df_quantiles = df_cross.quantile(q=quantiles).T
df_quantiles.columns = [f"{int(q*100)}%" if q != 0.50 else "Median" for q in quantiles]

tabla_8_jbf = pd.concat([df_mean, df_quantiles], axis=1)
tabla_8_jbf = tabla_8_jbf.loc[cols_objetivo]

print("\nTABLA: CROSS-SECTIONAL DISTRIBUTION")
print(tabla_8_jbf.round(4))

# 9. FRONTERA EFICIENTE
df_ef, opt_sharpe, opt_vol, mean_rets, cov_mat = get_efficient_frontier(arithmetic_returns, num_portfolios=50, risk_free_rate=0.0)
plot_efficient_frontier(df_ef, opt_sharpe, opt_vol, mean_rets, cov_mat, risk_free_rate=0.0)
