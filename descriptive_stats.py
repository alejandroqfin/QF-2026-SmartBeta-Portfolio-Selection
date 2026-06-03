"""
DESCRIPTIVE ANALYSIS: RETURNS, RISK METRICS & MARKET MODEL
Smart Beta ETF Universe - TFM
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
    kurtosis,
    get_efficient_frontier
)
from screening import SHARPE_RATIO, HISTORICAL_VAR
from plots import plot_boxplots, plot_efficient_frontier, plot_market_model_regression

# ---------------------------------------------------------
# 1. CARGA DE DATOS Y RENTABILIDADES
# ---------------------------------------------------------
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

log_returns = np.log(price_matrix / price_matrix.shift(1)).dropna(axis=0, how='all')
arithmetic_returns = price_matrix.pct_change().dropna(axis=0, how='all')
annual_factor = 252

# ---------------------------------------------------------
# 2. CÁLCULO DE MÉTRICAS Y RIESGO DE COLA
# ---------------------------------------------------------
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

df_metrics['VaRR_1'] = np.abs(df_metrics['VaR_01']) / np.abs(df_metrics['VaR_99']).replace(0, np.nan)
df_metrics['VaRR_5'] = np.abs(df_metrics['VaR_05']) / np.abs(df_metrics['VaR_95']).replace(0, np.nan)
df_metrics['VaRR_10'] = np.abs(df_metrics['VaR_10']) / np.abs(df_metrics['VaR_90']).replace(0, np.nan)

df_metrics['Ranking_Sharpe'] = df_metrics['Sharpe_Ratio'].rank(ascending=False)
df_metrics['Ranking_VaRR_5'] = df_metrics['VaRR_5'].rank(ascending=False)
df_metrics['Ranking_VaRR_1'] = df_metrics['VaRR_1'].rank(ascending=False)
df_metrics['Ranking_VaRR_10'] = df_metrics['VaRR_10'].rank(ascending=False)

spearman_corr_matrix = df_metrics[['Ranking_Sharpe', 'Ranking_VaRR_1', 'Ranking_VaRR_5', 'Ranking_VaRR_10']].corr(method='spearman')

plot_boxplots(df_metrics)
df_ef, opt_sharpe, opt_vol, mean_rets, cov_mat = get_efficient_frontier(arithmetic_returns, num_portfolios=50, risk_free_rate=0.0)
plot_efficient_frontier(df_ef, opt_sharpe, opt_vol, mean_rets, cov_mat)

# ---------------------------------------------------------
# 3. BENCHMARKS Y CORRELACIÓN
# ---------------------------------------------------------
start_date = arithmetic_returns.index.min().strftime('%Y-%m-%d')
end_date = arithmetic_returns.index.max().strftime('%Y-%m-%d')
print(f"\nDescargando Series de Mercado desde {start_date} hasta {end_date}")

sp500   = yf.download('^GSPC', start=start_date, end=end_date, progress=False, auto_adjust=False)['Close']
russell = yf.download('^RUT',  start=start_date, end=end_date, progress=False, auto_adjust=False)['Close']
crsp    = yf.download('VTI',   start=start_date, end=end_date, progress=False, auto_adjust=False)['Close']

df_benchmarks = pd.concat([
    pd.Series(sp500.squeeze(), name='SP500'), 
    pd.Series(russell.squeeze(), name='RUSSELL'), 
    pd.Series(crsp.squeeze(), name='CRSP')
], axis=1)

returns_benchmarks = df_benchmarks.pct_change().dropna()
returns_benchmarks.index = pd.to_datetime(returns_benchmarks.index).normalize()
arithmetic_returns.index = pd.to_datetime(arithmetic_returns.index).normalize()
df_regression = pd.concat([arithmetic_returns, returns_benchmarks], axis=1, join='inner').dropna()

print("\n" + "-" * 60)
print(" MATRIZ DE CORRELACIÓN DE BENCHMARKS ")
print("-" * 60)
corr_benchmarks = returns_benchmarks.corr()
print(corr_benchmarks.round(4))

# ---------------------------------------------------------
# 4. REGRESIONES OLS (UNIVARIANTES)
# ---------------------------------------------------------
etfs_internacionales = ["EFV", "DLS", "DEW", "DFE", "DFJ", "DIM", "DTH", "DOL", "DWM", "PXF", "PXH", "DNL", "EFG", "PIZ", "EFAV", "EEMV", "ACWV", "EELV", "FEM", "FEMS"]
df_regression = df_regression.drop(columns=etfs_internacionales, errors='ignore')
etfs_usa = [ticker for ticker in df_regression.columns if ticker not in ['SP500', 'RUSSELL', 'CRSP']]

lista_sp500   = []
lista_russell = []
lista_crsp    = []

X_sp500   = sm.add_constant(df_regression['SP500'])
X_russell = sm.add_constant(df_regression['RUSSELL'])
X_crsp    = sm.add_constant(df_regression['CRSP'])

for etf in etfs_usa:
    y = df_regression[etf]
    
    # Univariante S&P 500
    m_sp = sm.OLS(y, X_sp500).fit()
    lista_sp500.append({'ETF': etf, 'Alpha': m_sp.params['const'], 'Alpha_pval': m_sp.pvalues['const'], 'Beta': m_sp.params['SP500'], 'Beta_pval': m_sp.pvalues['SP500'], 'R2': m_sp.rsquared, 'Idiosincratico': m_sp.mse_resid, 'Predictions': m_sp.predict(X_sp500)})
    
    # Univariante Russell 2000
    m_rut = sm.OLS(y, X_russell).fit()
    lista_russell.append({'ETF': etf, 'Alpha': m_rut.params['const'], 'Alpha_pval': m_rut.pvalues['const'], 'Beta': m_rut.params['RUSSELL'], 'Beta_pval': m_rut.pvalues['RUSSELL'], 'R2': m_rut.rsquared, 'Idiosincratico': m_rut.mse_resid, 'Predictions': m_rut.predict(X_russell)})
    
    # Univariante CRSP
    m_crsp = sm.OLS(y, X_crsp).fit()
    lista_crsp.append({'ETF': etf, 'Alpha': m_crsp.params['const'], 'Alpha_pval': m_crsp.pvalues['const'], 'Beta': m_crsp.params['CRSP'], 'Beta_pval': m_crsp.pvalues['CRSP'], 'R2': m_crsp.rsquared, 'Idiosincratico': m_crsp.mse_resid, 'Predictions': m_crsp.predict(X_crsp)})


df_res_sp500 = pd.DataFrame(lista_sp500)
df_res_russell = pd.DataFrame(lista_russell)
df_res_crsp = pd.DataFrame(lista_crsp)

sig = 0.05
df_res_sp500['Alpha_Sig'] = df_res_sp500['Alpha_pval'] < sig
df_res_sp500['Beta_Sig'] = df_res_sp500['Beta_pval'] < sig
df_res_russell['Alpha_Sig'] = df_res_russell['Alpha_pval'] < sig
df_res_russell['Beta_Sig'] = df_res_russell['Beta_pval'] < sig
df_res_crsp['Alpha_Sig'] = df_res_crsp['Alpha_pval'] < sig
df_res_crsp['Beta_Sig'] = df_res_crsp['Beta_pval'] < sig

# ---------------------------------------------------------
# 5. TABLAS DESCRIPTIVAS
# ---------------------------------------------------------
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

tabla_sp500 = build_univariate_table(df_res_sp500)
tabla_russell = build_univariate_table(df_res_russell)
tabla_crsp = build_univariate_table(df_res_crsp)

print("\n TABLA: MODELO S&P 500")
print(tabla_sp500.round(4))
print("\n TABLA: MODELO RUSSELL 2000")
print(tabla_russell.round(4))
print("\n TABLA: MODELO CRSP TOTAL MARKET")
print(tabla_crsp.round(4))

# ---------------------------------------------------------
# 6. GRÁFICOS
# ---------------------------------------------------------
plot_market_model_regression(df_regression, df_res_sp500, index_col='SP500', index_name='S\&P 500', save_path='market_regression_sp500.pdf')
plot_market_model_regression(df_regression, df_res_russell, index_col='RUSSELL', index_name='Russell 2000', save_path='market_regression_russell.pdf')
plot_market_model_regression(df_regression, df_res_crsp, index_col='CRSP', index_name='CRSP Total Market', save_path='market_regression_crsp.pdf')

# Boxplots de las 3 Betas Univariantes Originales
df_betas_conjunto = pd.DataFrame({
    'SP500_Beta': df_res_sp500['Beta'].values,
    'RUSSELL_Beta': df_res_russell['Beta'].values,
    'CRSP_Beta': df_res_crsp['Beta'].values
})
plot_boxplots(df_betas_conjunto, is_beta_analysis=True, save_name="boxplots_betas_tfm.pdf")

# ---------------------------------------------------------
# 7. EXPORTACIÓN A EXCEL
# ---------------------------------------------------------
with pd.ExcelWriter(FILE_PATH_OUT, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    log_returns.to_excel(writer, sheet_name='Log Returns')
    arithmetic_returns.to_excel(writer, sheet_name='Arithmetic Returns')
    df_metrics.to_excel(writer, sheet_name='Risk Metrics')
    spearman_corr_matrix.to_excel(writer, sheet_name='Spearman')
    returns_benchmarks.to_excel(writer, sheet_name='Benchmarks Returns')
    corr_benchmarks.to_excel(writer, sheet_name='Benchmarks Corr')
    
    tabla_sp500.to_excel(writer, sheet_name='Market Model SP500')
    tabla_russell.to_excel(writer, sheet_name='Market Model Russell')
    tabla_crsp.to_excel(writer, sheet_name='Market Model CRSP')

print(f"\nMétricas calculadas y actualizadas en el archivo: {FILE_PATH_OUT}.")