"""
DESCRIPTIVE ANALYSIS: RETURNS, RISK METRICS & MARKET MODEL
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martínez
"""

import pandas as pd
import numpy as np
import shutil
import yfinance as yf
import statsmodels.api as sm
from metrics import (
    mean, 
    volatility, 
    skewness, 
    kurtosis
)

from screening import SHARPE_RATIO, HISTORICAL_VAR
from plots import plot_boxplots, plot_market_model_regression

# 1. CARGA DE DATOS Y RENTABILIDADES
FILE_PATH_IN = "100_ETFs_Smart_Beta.xlsx"
FILE_PATH_OUT = "100_ETFs_AnalisisDescriptivo.xlsx"
DATA_SHEET = "Precios_Historicos"
shutil.copy(FILE_PATH_IN, FILE_PATH_OUT)

price_matrix = pd.read_excel(
    FILE_PATH_IN, 
    sheet_name=DATA_SHEET, 
    index_col=0, 
    header=[0, 1], 
    parse_dates=True
)

price_matrix.columns = price_matrix.columns.get_level_values(1)
T, N = price_matrix.shape
print(f"Observaciones (T): {T} | ETFs (N): {N}")

# RENDIMIENTOS
log_returns = np.log(price_matrix / price_matrix.shift(1)).dropna(axis=0, how='all')
arithmetic_returns = price_matrix.pct_change().dropna(axis=0, how='all')
annual_factor = 252

# 2. CÁLCULO DE MÉTRICAS Y RIESGO DE COLA
df_metrics = pd.DataFrame({
    'Mean': mean(log_returns) * annual_factor,
    'Volatility': volatility(log_returns) * np.sqrt(annual_factor),
    'Skewness': skewness(log_returns),
    'Kurtosis': kurtosis(log_returns),
    'Sharpe': SHARPE_RATIO(log_returns, rf=0.0) * np.sqrt(annual_factor),
    'VaR_01': HISTORICAL_VAR(log_returns, tau=0.01),
    'VaR_05': HISTORICAL_VAR(log_returns, tau=0.05),
    'VaR_10': HISTORICAL_VAR(log_returns, tau=0.10),
    'VaR_90': HISTORICAL_VAR(log_returns, tau=0.90),
    'VaR_95': HISTORICAL_VAR(log_returns, tau=0.95), 
    'VaR_99': HISTORICAL_VAR(log_returns, tau=0.99) 
})

df_metrics['VaRR_1'] = np.abs(df_metrics['VaR_01']) / np.abs(df_metrics['VaR_99']).replace(0, np.nan)
df_metrics['VaRR_5'] = np.abs(df_metrics['VaR_05']) / np.abs(df_metrics['VaR_95']).replace(0, np.nan)
df_metrics['VaRR_10'] = np.abs(df_metrics['VaR_10']) / np.abs(df_metrics['VaR_90']).replace(0, np.nan)

df_metrics['Ranking_Sharpe'] = df_metrics['Sharpe'].rank(ascending=False)
df_metrics['Ranking_VaRR_5'] = df_metrics['VaRR_5'].rank(ascending=False)
df_metrics['Ranking_VaRR_1'] = df_metrics['VaRR_1'].rank(ascending=False)
df_metrics['Ranking_VaRR_10'] = df_metrics['VaRR_10'].rank(ascending=False)

# MATRIZ DE CORRELACIÓN DE SPEARMAN
rank_cols = ['Ranking_Sharpe', 'Ranking_VaRR_1', 'Ranking_VaRR_5', 'Ranking_VaRR_10']
short_names = ['Sharpe', 'VaRR(1%)', 'VaRR(5%)', 'VaRR(10%)']

spearman_corr_matrix = df_metrics[rank_cols].corr(method='spearman')
spearman_corr_matrix.columns = short_names
spearman_corr_matrix.index = short_names

print("\nMATRIZ DE CORRELACIÓN (SPEARMAN)")
print(spearman_corr_matrix.round(3))

# BOX-PLOTS
plot_boxplots(df_metrics)

# 3. ANÁLISIS DE MERCADO (BENCHMARKS)
# FECHAS
start_date = arithmetic_returns.index.min().strftime('%Y-%m-%d')
end_date = arithmetic_returns.index.max().strftime('%Y-%m-%d')
print(f"\nDescargando Series de Mercado desde {start_date} hasta {end_date}...")

# ÍNDICES DE MERCADO
sp500   = yf.download('^GSPC', start=start_date, end=end_date, progress=False, auto_adjust=False)['Close']
crsp    = yf.download('VTI',   start=start_date, end=end_date, progress=False, auto_adjust=False)['Close']

df_benchmarks = pd.concat([
    pd.Series(sp500.squeeze(), name='SP500'), 
    pd.Series(crsp.squeeze(), name='CRSP')
], axis=1)

returns_benchmarks = df_benchmarks.pct_change().dropna()
returns_benchmarks.index = pd.to_datetime(returns_benchmarks.index).normalize()
arithmetic_returns.index = pd.to_datetime(arithmetic_returns.index).normalize()
df_regression = pd.concat([arithmetic_returns, returns_benchmarks], axis=1, join='inner').dropna()

corr_value = returns_benchmarks['SP500'].corr(returns_benchmarks['CRSP'])
print(f"\nCorrelación entre S&P 500 y CRSP Total Market: {corr_value * 100:.2f}%")

# 4. REGRESIÓN POR MÍNIMOS CUADRADOS ORDINARIOS (MCO)
etfs_internacionales = ["EFV", "DLS", "DEW", "DFE", "DFJ", "DIM", "DTH", "DOL", "DWM", "PXF", "PXH", "DNL", "EFG", "PIZ", "EFAV", "EEMV", "ACWV", "EELV", "FEM", "FEMS"]
df_regression = df_regression.drop(columns=etfs_internacionales, errors='ignore')
etfs_usa = [ticker for ticker in df_regression.columns if ticker not in ['SP500', 'RUSSELL', 'CRSP']]

lista_sp500   = []
lista_crsp    = []

X_sp500   = sm.add_constant(df_regression['SP500'])
X_crsp    = sm.add_constant(df_regression['CRSP'])

for etf in etfs_usa:
    y = df_regression[etf]
    
    # S&P 500
    m_sp = sm.OLS(y, X_sp500).fit()
    lista_sp500.append({'ETF': etf, 'Alpha': m_sp.params['const'], 'Alpha_pval': m_sp.pvalues['const'], 'Beta': m_sp.params['SP500'], 'Beta_pval': m_sp.pvalues['SP500'], 'R2': m_sp.rsquared, 'Idiosincratico': m_sp.mse_resid, 'Predictions': m_sp.predict(X_sp500)})
    
    # CRSP
    m_crsp = sm.OLS(y, X_crsp).fit()
    lista_crsp.append({'ETF': etf, 'Alpha': m_crsp.params['const'], 'Alpha_pval': m_crsp.pvalues['const'], 'Beta': m_crsp.params['CRSP'], 'Beta_pval': m_crsp.pvalues['CRSP'], 'R2': m_crsp.rsquared, 'Idiosincratico': m_crsp.mse_resid, 'Predictions': m_crsp.predict(X_crsp)})

df_res_sp500 = pd.DataFrame(lista_sp500)
df_res_crsp = pd.DataFrame(lista_crsp)

# ALPHA
sig = 0.05
df_res_sp500['Alpha_Sig'] = df_res_sp500['Alpha_pval'] < sig
df_res_sp500['Beta_Sig'] = df_res_sp500['Beta_pval'] < sig
df_res_crsp['Alpha_Sig'] = df_res_crsp['Alpha_pval'] < sig
df_res_crsp['Beta_Sig'] = df_res_crsp['Beta_pval'] < sig

# 5. TABLAS DESCRIPTIVAS
def build_univariate_table(df):
    resumen = {
        'Métrica': ['Alpha', 'Beta', 'R2', 'Riesgo Idiosincrático'],
        'Mean': [df['Alpha'].mean() * annual_factor * 100, df['Beta'].mean(), df['R2'].mean() * 100, df['Idiosincratico'].mean() * annual_factor],
        'Q1 (25%)': [df['Alpha'].quantile(0.25) * annual_factor * 100, df['Beta'].quantile(0.25), df['R2'].quantile(0.25) * 100, df['Idiosincratico'].quantile(0.25) * annual_factor],
        'Median': [df['Alpha'].median() * annual_factor * 100, df['Beta'].median(), df['R2'].median() * 100, df['Idiosincratico'].median() * annual_factor],
        'Q3 (75%)': [df['Alpha'].quantile(0.75) * annual_factor * 100, df['Beta'].quantile(0.75), df['R2'].quantile(0.75) * 100, df['Idiosincratico'].quantile(0.75) * annual_factor],
        'Std Error': [df['Alpha'].sem() * annual_factor * 100, df['Beta'].sem(), df['R2'].sem() * 100, df['Idiosincratico'].sem() * annual_factor],
        'Significativos (p < 0.05)': [f"{df['Alpha_Sig'].sum()} / {len(df)}", f"{df['Beta_Sig'].sum()} / {len(df)}", "--", "--"]
    }
    return pd.DataFrame(resumen).set_index('Métrica')

print("\n TABLA: MODELO S&P 500")
tabla_sp500 = build_univariate_table(df_res_sp500)
print(tabla_sp500.round(4))

print("\n TABLA: MODELO CRSP TOTAL MARKET")
tabla_crsp = build_univariate_table(df_res_crsp)
print(tabla_crsp.round(4))

# 6. GRÁFICOS
plot_market_model_regression(df_regression, df_res_sp500, index_col='SP500', index_name='S\&P 500', save_path='market_regression_sp500.pdf')
plot_market_model_regression(df_regression, df_res_crsp, index_col='CRSP', index_name='CRSP Total Market', save_path='market_regression_crsp.pdf')

# BOXPLOTS
df_betas_conjunto = pd.DataFrame({
    'SP500_Beta': df_res_sp500['Beta'].values,
    'CRSP_Beta': df_res_crsp['Beta'].values
})

# 7. EXPORTACIÓN A EXCEL
with pd.ExcelWriter(FILE_PATH_OUT, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    
    # RENDIMIENTOS
    log_returns.to_excel(writer, sheet_name='Log Returns')
    arithmetic_returns.to_excel(writer, sheet_name='Arithmetic Returns')
    returns_benchmarks.to_excel(writer, sheet_name='Benchmarks Returns')
    
    # ESTADÍSTICOS DE RIESGO
    df_metrics.to_excel(writer, sheet_name='Risk Metrics')
    spearman_corr_matrix.to_excel(writer, sheet_name='Spearman')
    
    # MODELOS DE MERCADO
    tabla_sp500.to_excel(writer, sheet_name='Market Model SP500')
    tabla_crsp.to_excel(writer, sheet_name='Market Model CRSP')

print(f"\nMétricas calculadas y exportadas en el archivo: {FILE_PATH_OUT}.")