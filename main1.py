"""
main1.py
PARTE I: ROLLING WINDOW RANKINGS (SELECCIÓN DINÁMICA DE CARTERAS)
Smart Beta ETF Universe - TFM
Autor: Alejandro Martínez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from screening import ROLLING_WINDOW_RANKINGS
from metrics import etf_stats, replacement_rate, transaction_costs
from plots import plot_cumulative_returns, plot_turnover_frequency

INPUT_FILE = "100_ETFs_Smart_Beta.xlsx"
OUTPUT_FILE = "100_ETFs_Screening.xlsx"
VARIABLES_FILE = "screening_artifacts.joblib"
OOS_START_DATE = pd.to_datetime('2023-01-03')

# ETFs EN CARTERA
K = 20

# VENTANA IN SAMPLE
M = 1000

# EXTRAEMOS PRECIOS (P_T): (M + T X 100)
df_precios = pd.read_excel(INPUT_FILE, sheet_name='Precios_Historicos', index_col=0, header=1)
df_precios = df_precios.iloc[1:]
df_precios.index = pd.to_datetime(df_precios.index).normalize()
df_precios = df_precios.astype(float)

# RENDIMIENTOS ARITMÉTICOS (P_T / P_T-1 - 1): (M + T - 1 X 100)
df_rendimientos = df_precios.pct_change().dropna(how='all')

# ASIGNAMOS UN ID NUMÉRICO A CADA ETF (DEL 1 A 100)
tickers = df_rendimientos.columns.tolist()
ticker_to_id = {ticker: idx + 1 for idx, ticker in enumerate(tickers)}
id_to_ticker = {v: k for k, v in ticker_to_id.items()}

df_rendimientos.columns = [ticker_to_id[ticker] for ticker in tickers]
df_ID = pd.DataFrame({
    'ID': list(ticker_to_id.values()),
    'ETF': list(ticker_to_id.keys())
})

# DATOS
print("\nDatos:")
print(f"   • Universo de ETFs:            {len(tickers)}")
print(f"   • ETFs en cartera:             {K}")
print(f"   • Días totales:                {len(df_rendimientos)}")
print(f"   • Días IS:                     {len(df_rendimientos[df_rendimientos.index < OOS_START_DATE])}")
print(f"   • Días OOS:                    {len(df_rendimientos[df_rendimientos.index >= OOS_START_DATE])}")
print(f"   • Tamaño de la ventana IS:     {M} días")
print(f"   • Fecha inicial:               {df_rendimientos.index.min().date()}")
print(f"   • Fecha final:                 {df_rendimientos.index.max().date()}")
print(f"   • Fecha inicial OOS:           {OOS_START_DATE.date()}")
print(f"   • NaNs (%):                    {100.0 * df_rendimientos.isna().sum().sum() / df_rendimientos.size:.4f}")

# PARTE 1: ROLLING WINDOW
ratios = ['SR', 'SR_corr', 'MSR', 'VaRR1', 'VaRR5', 'VaRR10', 'Omega', 'UPR', 'Kappa3', 'Sortino', 'GR1', 'GR5']
rankings_oos, series_oos = ROLLING_WINDOW_RANKINGS(df_rendimientos, OOS_START_DATE, K, ratios, M=M)

# RANKINGS OOS (FECHAS DESDE OOS_START_DATE HASTA EL FINAL)2
ranking_SR_oos = rankings_oos['SR']
ranking_SR_corr_oos = rankings_oos['SR_corr']
ranking_MSR_oos = rankings_oos['MSR']
ranking_VaRR1_oos = rankings_oos['VaRR1']
ranking_VaRR5_oos = rankings_oos['VaRR5']
ranking_VaRR10_oos = rankings_oos['VaRR10']
ranking_Omega_oos = rankings_oos['Omega']
ranking_UPR_oos = rankings_oos['UPR']
ranking_Kappa3_oos = rankings_oos['Kappa3']
ranking_Sortino_oos = rankings_oos['Sortino']
ranking_GR1_oos = rankings_oos['GR1']
ranking_GR5_oos = rankings_oos['GR5']

# SERIES OOS DE LOS RATIOS (FECHAS DESDE OOS_START_DATE HASTA EL FINAL)
serie_SR_oos = series_oos['SR']
serie_SR_corr_oos = series_oos['SR_corr']
serie_MSR_oos = series_oos['MSR']
serie_VaRR1_oos = series_oos['VaRR1']
serie_VaRR5_oos = series_oos['VaRR5']
serie_VaRR10_oos = series_oos['VaRR10']
serie_Omega_oos = series_oos['Omega']
serie_UPR_oos = series_oos['UPR']
serie_Kappa3_oos = series_oos['Kappa3']
serie_Sortino_oos = series_oos['Sortino']
serie_GR1_oos = series_oos['GR1']
serie_GR5_oos = series_oos['GR5']

# PARTE 2: COMPARACIÓN ENTRE CARTERAS
# VECTOR DE PESOS EQUIPONDERADOS
w = np.full(K, 1 / K)

# 2A. SIMULACIÓN DE CARTERA ROLLING (DINÁMICA)
# PERIODO OUT OF SAMPLE (T)
T = df_rendimientos.index[df_rendimientos.index >= OOS_START_DATE]
t = ranking_SR_oos.index.get_indexer(T)
t_anterior = np.maximum(0, t - 1)

# MATRICES DE IDS SELECCIONADOS: (T X K)
ids_rolling_SR = (ranking_SR_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_SR_corr = (ranking_SR_corr_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_MSR = (ranking_MSR_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_VaRR5 = (ranking_VaRR5_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_VaRR1 = (ranking_VaRR1_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_VaRR10 = (ranking_VaRR10_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_Omega = (ranking_Omega_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_UPR = (ranking_UPR_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_Kappa3 = (ranking_Kappa3_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_Sortino = (ranking_Sortino_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_GR1 = (ranking_GR1_oos.iloc[t_anterior].values - 1).astype(int)
ids_rolling_GR5 = (ranking_GR5_oos.iloc[t_anterior].values - 1).astype(int)

# EXTRAEMOS LA SUBMATRIZ DE RENDIMIENTOS OOS: (T X 100)
R_oos = df_rendimientos.loc[T].values

# VECTOR COLUMNA (T X 1) DE TIEMPO, DESDE 0 HASTA T-1
t_oos = np.arange(len(T))[:, None]

# RENDIMIENTOS INDIVIDUALES DE LOS K MEJORES ETFS PARA CADA T
# SUBMATRIZ: (T X K)
R_rolling_SR = R_oos[t_oos, ids_rolling_SR]
R_rolling_SR_corr = R_oos[t_oos, ids_rolling_SR_corr]
R_rolling_MSR = R_oos[t_oos, ids_rolling_MSR]
R_rolling_VaRR1 = R_oos[t_oos, ids_rolling_VaRR1]
R_rolling_VaRR5 = R_oos[t_oos, ids_rolling_VaRR5]
R_rolling_VaRR10 = R_oos[t_oos, ids_rolling_VaRR10]
R_rolling_Omega = R_oos[t_oos, ids_rolling_Omega]
R_rolling_UPR = R_oos[t_oos, ids_rolling_UPR]
R_rolling_Kappa3 = R_oos[t_oos, ids_rolling_Kappa3]
R_rolling_Sortino = R_oos[t_oos, ids_rolling_Sortino]
R_rolling_GR1 = R_oos[t_oos, ids_rolling_GR1]
R_rolling_GR5 = R_oos[t_oos, ids_rolling_GR5]

# RENDIMIENTOS DE LAS CARTERAS ROLLING EQUIPONDERADAS
# PRODUCTO MATRICIAL (SUMATORIO): R_P = R_I @ W
# DIMENSIONES: (T X 1) = (T X K) @ (K X 1)
R_cartera_SR = pd.Series(R_rolling_SR @ w, index=T)
R_cartera_SR_corr = pd.Series(R_rolling_SR_corr @ w, index=T)
R_cartera_MSR = pd.Series(R_rolling_MSR @ w, index=T)
R_cartera_VaRR1 = pd.Series(R_rolling_VaRR1 @ w, index=T)
R_cartera_VaRR5 = pd.Series(R_rolling_VaRR5 @ w, index=T)
R_cartera_VaRR10 = pd.Series(R_rolling_VaRR10 @ w, index=T)
R_cartera_Omega = pd.Series(R_rolling_Omega @ w, index=T)
R_cartera_UPR = pd.Series(R_rolling_UPR @ w, index=T)
R_cartera_Kappa3 = pd.Series(R_rolling_Kappa3 @ w, index=T)
R_cartera_Sortino = pd.Series(R_rolling_Sortino @ w, index=T)
R_cartera_GR1 = pd.Series(R_rolling_GR1 @ w, index=T)
R_cartera_GR5 = pd.Series(R_rolling_GR5 @ w, index=T)

# INVERSIÓN INICIAL DE 1€ 
W0 = 1

# RIQUEZA ACUMULADA DE LA CARTERA ROLLING (PRODUCTORIO): 
# WT = W0 PROD(1 + RP,T) DESDE T=1 HASTA T
wealth_rolling_SR = W0 * (1 + R_cartera_SR).cumprod()
wealth_rolling_SR_corr = W0 * (1 + R_cartera_SR_corr).cumprod()
wealth_rolling_MSR = W0 * (1 + R_cartera_MSR).cumprod()
wealth_rolling_VaRR1 = W0 * (1 + R_cartera_VaRR1).cumprod()
wealth_rolling_VaRR5 = W0 * (1 + R_cartera_VaRR5).cumprod()
wealth_rolling_VaRR10 = W0 * (1 + R_cartera_VaRR10).cumprod()
wealth_rolling_Omega = W0 * (1 + R_cartera_Omega).cumprod()
wealth_rolling_UPR = W0 * (1 + R_cartera_UPR).cumprod()
wealth_rolling_Kappa3 = W0 * (1 + R_cartera_Kappa3).cumprod()
wealth_rolling_Sortino = W0 * (1 + R_cartera_Sortino).cumprod()
wealth_rolling_GR1 = W0 * (1 + R_cartera_GR1).cumprod()
wealth_rolling_GR5 = W0 * (1 + R_cartera_GR5).cumprod()

# 2B. CARTERA BUY & HOLD PURA
# SOLO INVERTIMOS EN T = 0 Y NO INTERVENIMOS MÁS
t0 = ranking_SR_oos.index[ranking_SR_oos.index < OOS_START_DATE][-1]

# K MEJORES ETFS EN T=0 (Y LOS CONGELAMOS): (1 X K)
ids_buyhold_SR = ranking_SR_oos.loc[t0].values
ids_buyhold_SR_corr = ranking_SR_corr_oos.loc[t0].values
ids_buyhold_MSR = ranking_MSR_oos.loc[t0].values
ids_buyhold_VaRR1 = ranking_VaRR1_oos.loc[t0].values
ids_buyhold_VaRR5 = ranking_VaRR5_oos.loc[t0].values
ids_buyhold_VaRR10 = ranking_VaRR10_oos.loc[t0].values
ids_buyhold_Omega = ranking_Omega_oos.loc[t0].values
ids_buyhold_UPR = ranking_UPR_oos.loc[t0].values
ids_buyhold_Kappa3 = ranking_Kappa3_oos.loc[t0].values
ids_buyhold_Sortino = ranking_Sortino_oos.loc[t0].values
ids_buyhold_GR1 = ranking_GR1_oos.loc[t0].values
ids_buyhold_GR5 = ranking_GR5_oos.loc[t0].values

# MATRIZ DE RENDIMIENTOS OOS DE LOS ETFS SELECCIONADOS POR LA B&H: (T X K)
R_buyhold_SR = df_rendimientos.loc[T, ids_buyhold_SR]
R_buyhold_SR_corr = df_rendimientos.loc[T, ids_buyhold_SR_corr]
R_buyhold_MSR = df_rendimientos.loc[T, ids_buyhold_MSR]
R_buyhold_VaRR1 = df_rendimientos.loc[T, ids_buyhold_VaRR1]
R_buyhold_VaRR5 = df_rendimientos.loc[T, ids_buyhold_VaRR5]
R_buyhold_VaRR10 = df_rendimientos.loc[T, ids_buyhold_VaRR10]
R_buyhold_Omega = df_rendimientos.loc[T, ids_buyhold_Omega]
R_buyhold_UPR = df_rendimientos.loc[T, ids_buyhold_UPR]
R_buyhold_Kappa3 = df_rendimientos.loc[T, ids_buyhold_Kappa3]
R_buyhold_Sortino = df_rendimientos.loc[T, ids_buyhold_Sortino]
R_buyhold_GR1 = df_rendimientos.loc[T, ids_buyhold_GR1]
R_buyhold_GR5 = df_rendimientos.loc[T, ids_buyhold_GR5]

# RIQUEZA ACUMULADA (T X K)
etf_wealth_buyhold_SR = W0 * (1 + R_buyhold_SR).cumprod()
etf_wealth_buyhold_SR_corr = W0 * (1 + R_buyhold_SR_corr).cumprod()
etf_wealth_buyhold_MSR = W0 * (1 + R_buyhold_MSR).cumprod()
etf_wealth_buyhold_VaRR1 = W0 * (1 + R_buyhold_VaRR1).cumprod()
etf_wealth_buyhold_VaRR5 = W0 * (1 + R_buyhold_VaRR5).cumprod() 
etf_wealth_buyhold_VaRR10 = W0 * (1 + R_buyhold_VaRR10).cumprod()
etf_wealth_buyhold_Omega = W0 * (1 + R_buyhold_Omega).cumprod()
etf_wealth_buyhold_UPR = W0 * (1 + R_buyhold_UPR).cumprod()
etf_wealth_buyhold_Kappa3 = W0 * (1 + R_buyhold_Kappa3).cumprod()
etf_wealth_buyhold_Sortino = W0 * (1 + R_buyhold_Sortino).cumprod()
etf_wealth_buyhold_GR1 = W0 * (1 + R_buyhold_GR1).cumprod()
etf_wealth_buyhold_GR5 = W0 * (1 + R_buyhold_GR5).cumprod()

# SUMA PONDERADA DE LA RIQUEZA PATRIMONIAL POR ETF
# DIMENSIONES: (T X K) @ (K X 1) = (T X 1)
wealth_buyhold_SR = pd.Series(etf_wealth_buyhold_SR.values @ w, index=T)
wealth_buyhold_SR_corr = pd.Series(etf_wealth_buyhold_SR_corr.values @ w, index=T)
wealth_buyhold_MSR = pd.Series(etf_wealth_buyhold_MSR.values @ w, index=T)
wealth_buyhold_VaRR1 = pd.Series(etf_wealth_buyhold_VaRR1.values @ w, index=T)
wealth_buyhold_VaRR5 = pd.Series(etf_wealth_buyhold_VaRR5.values @ w, index=T)
wealth_buyhold_VaRR10 = pd.Series(etf_wealth_buyhold_VaRR10.values @ w, index=T)
wealth_buyhold_Omega = pd.Series(etf_wealth_buyhold_Omega.values @ w, index=T)
wealth_buyhold_UPR = pd.Series(etf_wealth_buyhold_UPR.values @ w, index=T)
wealth_buyhold_Kappa3 = pd.Series(etf_wealth_buyhold_Kappa3.values @ w, index=T)
wealth_buyhold_Sortino = pd.Series(etf_wealth_buyhold_Sortino.values @ w, index=T)
wealth_buyhold_GR1 = pd.Series(etf_wealth_buyhold_GR1.values @ w, index=T)
wealth_buyhold_GR5 = pd.Series(etf_wealth_buyhold_GR5.values @ w, index=T)

# PARTE 3: CALCULAMOS EL NÚMERO DE ETFS QUE ENTRAN CADA DÍA A LA CARTERA
replacement_rate_SR, freq_SR = replacement_rate(ranking_SR_oos, T, return_frequency=True)
replacement_rate_SR_corr, freq_SR_corr = replacement_rate(ranking_SR_corr_oos, T, return_frequency=True)
replacement_rate_MSR, freq_MSR = replacement_rate(ranking_MSR_oos, T, return_frequency=True)
replacement_rate_VaRR1, freq_VaRR1 = replacement_rate(ranking_VaRR1_oos, T, return_frequency=True)
replacement_rate_VaRR5, freq_VaRR5 = replacement_rate(ranking_VaRR5_oos, T, return_frequency=True)
replacement_rate_VaRR10, freq_VaRR10 = replacement_rate(ranking_VaRR10_oos, T, return_frequency=True)
replacement_rate_Omega, freq_Omega = replacement_rate(ranking_Omega_oos, T, return_frequency=True)
replacement_rate_UPR, freq_UPR = replacement_rate(ranking_UPR_oos, T, return_frequency=True)
replacement_rate_Kappa3, freq_Kappa3 = replacement_rate(ranking_Kappa3_oos, T, return_frequency=True)
replacement_rate_Sortino, freq_Sortino = replacement_rate(ranking_Sortino_oos, T, return_frequency=True)
replacement_rate_GR1, freq_GR1 = replacement_rate(ranking_GR1_oos, T, return_frequency=True)
replacement_rate_GR5, freq_GR5 = replacement_rate(ranking_GR5_oos, T, return_frequency=True)

# TABLA DE REEMPLAZOS DIARIOS DE ETFS EN CARTERA (0 - 1 - 2 ... - K)
df_turnover_activos = pd.DataFrame({
    'Nuevos_SR': replacement_rate_SR,
    'Nuevos_SR_corr': replacement_rate_SR_corr,
    'Nuevos_MSR': replacement_rate_MSR,
    'Nuevos_VaRR1': replacement_rate_VaRR1,
    'Nuevos_VaRR5': replacement_rate_VaRR5,
    'Nuevos_VaRR10': replacement_rate_VaRR10,
    'Nuevos_Omega': replacement_rate_Omega,
    'Nuevos_UPR': replacement_rate_UPR,
    'Nuevos_Kappa3': replacement_rate_Kappa3,
    'Nuevos_Sortino': replacement_rate_Sortino,
    'Nuevos_GR1': replacement_rate_GR1,
    'Nuevos_GR5': replacement_rate_GR5,
})

print("\nTasa de rebalanceo diaria de ETFs:")
print(f"   • Sharpe              : {df_turnover_activos['Nuevos_SR'].mean():.4f} de {K} ETFs por día")
print(f"   • Sharpe Correlation  : {df_turnover_activos['Nuevos_SR_corr'].mean():.4f} de {K} ETFs por día")
print(f"   • MSR                 : {df_turnover_activos['Nuevos_MSR'].mean():.4f} de {K} ETFs por día")
print(f"   • VaRR1               : {df_turnover_activos['Nuevos_VaRR1'].mean():.4f} de {K} ETFs por día")
print(f"   • VaRR5               : {df_turnover_activos['Nuevos_VaRR5'].mean():.4f} de {K} ETFs por día")
print(f"   • VaRR10              : {df_turnover_activos['Nuevos_VaRR10'].mean():.4f} de {K} ETFs por día")
print(f"   • Omega               : {df_turnover_activos['Nuevos_Omega'].mean():.4f} de {K} ETFs por día")
print(f"   • UPR                 : {df_turnover_activos['Nuevos_UPR'].mean():.4f} de {K} ETFs por día")
print(f"   • Kappa3              : {df_turnover_activos['Nuevos_Kappa3'].mean():.4f} de {K} ETFs por día")
print(f"   • Sortino             : {df_turnover_activos['Nuevos_Sortino'].mean():.4f} de {K} ETFs por día")
print(f"   • GR1                 : {df_turnover_activos['Nuevos_GR1'].mean():.4f} de {K} ETFs por día")
print(f"   • GR5                 : {df_turnover_activos['Nuevos_GR5'].mean():.4f} de {K} ETFs por día")

# TABLA DE FRECUENCIAS DE REBALANCEO (0 - 1 - 2 ... - K)
df_turnover_freq = pd.DataFrame({
    'SR': freq_SR,
    'SR_corr': freq_SR_corr,
    'MSR': freq_MSR,
    'VaRR1': freq_VaRR1,
    'VaRR5': freq_VaRR5,
    'VaRR10': freq_VaRR10,
    'Omega': freq_Omega,
    'UPR': freq_UPR,
    'Kappa3': freq_Kappa3,
    'Sortino': freq_Sortino,
    'GR1': freq_GR1,
    'GR5': freq_GR5,
})

print("\nFrecuencia de rebalanceo (Número de ETFs nuevos en cartera):")

# 1. SHARPE RATIO
print("\n   ► Sharpe Ratio :")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['SR'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['SR'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['SR'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['SR'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['SR'].get(4, 0)} veces")

# 2. SHARPE RATIO CORRELATION
print("\n   ► Sharpe Correlation :")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['SR_corr'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['SR_corr'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['SR_corr'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['SR_corr'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['SR_corr'].get(4, 0)} veces")

# 3. MARGINAL SHARPE RATIO
print("\n   ► Marginal Sharpe Ratio (MSR) :")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['MSR'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['MSR'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['MSR'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['MSR'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['MSR'].get(4, 0)} veces")

# 4. VAR RATIO 1%
print("\n   ► VaR Ratio 1% :")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['VaRR1'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['VaRR1'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['VaRR1'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['VaRR1'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['VaRR1'].get(4, 0)} veces")

# 5. VAR RATIO 5%
print("\n   ► VaR Ratio 5% :")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['VaRR5'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['VaRR5'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['VaRR5'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['VaRR5'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['VaRR5'].get(4, 0)} veces")

# 6. VAR RATIO 10%
print("\n   ► VaR Ratio 10%:")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['VaRR10'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['VaRR10'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['VaRR10'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['VaRR10'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['VaRR10'].get(4, 0)} veces")

# 7. OMEGA
print("\n   ► Omega:")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['Omega'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['Omega'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['Omega'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['Omega'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['Omega'].get(4, 0)} veces")

# 8. UPR
print("\n   ► UPR:")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['UPR'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['UPR'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['UPR'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['UPR'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['UPR'].get(4, 0)} veces")

# 9. KAPPA3
print("\n   ► Kappa3:")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['Kappa3'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['Kappa3'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['Kappa3'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['Kappa3'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['Kappa3'].get(4, 0)} veces")

# 10. SORTINO
print("\n   ► Sortino:")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['Sortino'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['Sortino'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['Sortino'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['Sortino'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['Sortino'].get(4, 0)} veces")

# 11. GR1
print("\n   ► GR1:")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['GR1'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['GR1'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['GR1'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['GR1'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['GR1'].get(4, 0)} veces")

# 12. GR5
print("\n   ► GR5:")
print(f"      • 0 ETFs nuevos: {df_turnover_freq['GR5'].get(0, 0)} veces")
print(f"      • 1 ETFs nuevos: {df_turnover_freq['GR5'].get(1, 0)} veces")
print(f"      • 2 ETFs nuevos: {df_turnover_freq['GR5'].get(2, 0)} veces")
print(f"      • 3 ETFs nuevos: {df_turnover_freq['GR5'].get(3, 0)} veces")
print(f"      • 4 ETFs nuevos: {df_turnover_freq['GR5'].get(4, 0)} veces")

# GRÁFICO DE FRECUENCIA DE REBALANCEO
plot_turnover_frequency({
        'SR': freq_SR,
        'Sharpe Correlation': freq_SR_corr,
        'MSR': freq_MSR,
        'VaRR1': freq_VaRR1,
        'VaRR5': freq_VaRR5,
        'VaRR10': freq_VaRR10,
        'Omega': freq_Omega,
        'UPR': freq_UPR,
        'Kappa3': freq_Kappa3,
        'Sortino': freq_Sortino,
        'GR1': freq_GR1,
        'GR5': freq_GR5}
)

# ESTADÍSTICAS DE APARICIÓN Y POSICIÓN DE ETFS
stats_SR = etf_stats(ids_rolling_SR, tickers)
stats_SR_corr = etf_stats(ids_rolling_SR_corr, tickers)
stats_MSR = etf_stats(ids_rolling_MSR, tickers)
stats_VaRR1 = etf_stats(ids_rolling_VaRR1, tickers)
stats_VaRR5 = etf_stats(ids_rolling_VaRR5, tickers)
stats_VaRR10 = etf_stats(ids_rolling_VaRR10, tickers)
stats_Omega = etf_stats(ids_rolling_Omega, tickers)
stats_UPR = etf_stats(ids_rolling_UPR, tickers)
stats_Kappa3 = etf_stats(ids_rolling_Kappa3, tickers)
stats_Sortino = etf_stats(ids_rolling_Sortino, tickers)
stats_GR1 = etf_stats(ids_rolling_GR1, tickers)
stats_GR5 = etf_stats(ids_rolling_GR5, tickers)

# ORDENAMOS DE MENOR A MAYOR
stats_SR = stats_SR.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_SR_corr = stats_SR_corr.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_MSR = stats_MSR.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_VaRR1 = stats_VaRR1.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_VaRR5 = stats_VaRR5.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_VaRR10 = stats_VaRR10.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_Omega = stats_Omega.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_UPR = stats_UPR.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_Kappa3 = stats_Kappa3.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_Sortino = stats_Sortino.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_GR1 = stats_GR1.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)
stats_GR5 = stats_GR5.sort_values(by='Days_Active', ascending=False).reset_index(drop=True)

print("\nTop 5 ETFs que más tiempo han permanecido en cartera:")

# 1. SHARPE RATIO
print("\n   ► Sharpe Ratio:")
print(f"      1. {stats_SR.iloc[0]['Ticker']:<10} ({stats_SR.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_SR.iloc[1]['Ticker']:<10} ({stats_SR.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_SR.iloc[2]['Ticker']:<10} ({stats_SR.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_SR.iloc[3]['Ticker']:<10} ({stats_SR.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_SR.iloc[4]['Ticker']:<10} ({stats_SR.iloc[4]['Days_Active']} días)")

# 2. SHARPE CORRELATION
print("\n   ► Sharpe Correlation:")
print(f"      1. {stats_SR_corr.iloc[0]['Ticker']:<10} ({stats_SR_corr.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_SR_corr.iloc[1]['Ticker']:<10} ({stats_SR_corr.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_SR_corr.iloc[2]['Ticker']:<10} ({stats_SR_corr.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_SR_corr.iloc[3]['Ticker']:<10} ({stats_SR_corr.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_SR_corr.iloc[4]['Ticker']:<10} ({stats_SR_corr.iloc[4]['Days_Active']} días)")

# 3. MARGINAL SHARPE RATIO
print("\n   ► Marginal Sharpe Ratio (MSR):")
print(f"      1. {stats_MSR.iloc[0]['Ticker']:<10} ({stats_MSR.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_MSR.iloc[1]['Ticker']:<10} ({stats_MSR.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_MSR.iloc[2]['Ticker']:<10} ({stats_MSR.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_MSR.iloc[3]['Ticker']:<10} ({stats_MSR.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_MSR.iloc[4]['Ticker']:<10} ({stats_MSR.iloc[4]['Days_Active']} días)")

# 4. VAR RATIO 1%
print("\n   ► VaR Ratio 1%:")
print(f"      1. {stats_VaRR1.iloc[0]['Ticker']:<10} ({stats_VaRR1.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_VaRR1.iloc[1]['Ticker']:<10} ({stats_VaRR1.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_VaRR1.iloc[2]['Ticker']:<10} ({stats_VaRR1.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_VaRR1.iloc[3]['Ticker']:<10} ({stats_VaRR1.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_VaRR1.iloc[4]['Ticker']:<10} ({stats_VaRR1.iloc[4]['Days_Active']} días)")

# 5. VAR RATIO 5%
print("\n   ► VaR Ratio 5%:")
print(f"      1. {stats_VaRR5.iloc[0]['Ticker']:<10} ({stats_VaRR5.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_VaRR5.iloc[1]['Ticker']:<10} ({stats_VaRR5.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_VaRR5.iloc[2]['Ticker']:<10} ({stats_VaRR5.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_VaRR5.iloc[3]['Ticker']:<10} ({stats_VaRR5.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_VaRR5.iloc[4]['Ticker']:<10} ({stats_VaRR5.iloc[4]['Days_Active']} días)")

# 6. VAR RATIO 10%
print("\n   ► VaR Ratio 10%:")
print(f"      1. {stats_VaRR10.iloc[0]['Ticker']:<10} ({stats_VaRR10.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_VaRR10.iloc[1]['Ticker']:<10} ({stats_VaRR10.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_VaRR10.iloc[2]['Ticker']:<10} ({stats_VaRR10.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_VaRR10.iloc[3]['Ticker']:<10} ({stats_VaRR10.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_VaRR10.iloc[4]['Ticker']:<10} ({stats_VaRR10.iloc[4]['Days_Active']} días)")

# 7. OMEGA
print("\n   ► Omega:")
print(f"      1. {stats_Omega.iloc[0]['Ticker']:<10} ({stats_Omega.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_Omega.iloc[1]['Ticker']:<10} ({stats_Omega.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_Omega.iloc[2]['Ticker']:<10} ({stats_Omega.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_Omega.iloc[3]['Ticker']:<10} ({stats_Omega.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_Omega.iloc[4]['Ticker']:<10} ({stats_Omega.iloc[4]['Days_Active']} días)")

# 8. UPR
print("\n   ► UPR:")
print(f"      1. {stats_UPR.iloc[0]['Ticker']:<10} ({stats_UPR.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_UPR.iloc[1]['Ticker']:<10} ({stats_UPR.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_UPR.iloc[2]['Ticker']:<10} ({stats_UPR.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_UPR.iloc[3]['Ticker']:<10} ({stats_UPR.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_UPR.iloc[4]['Ticker']:<10} ({stats_UPR.iloc[4]['Days_Active']} días)")

# 9. KAPPA3
print("\n   ► Kappa3:")
print(f"      1. {stats_Kappa3.iloc[0]['Ticker']:<10} ({stats_Kappa3.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_Kappa3.iloc[1]['Ticker']:<10} ({stats_Kappa3.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_Kappa3.iloc[2]['Ticker']:<10} ({stats_Kappa3.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_Kappa3.iloc[3]['Ticker']:<10} ({stats_Kappa3.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_Kappa3.iloc[4]['Ticker']:<10} ({stats_Kappa3.iloc[4]['Days_Active']} días)")

# 10. SORTINO
print("\n   ► Sortino:")
print(f"      1. {stats_Sortino.iloc[0]['Ticker']:<10} ({stats_Sortino.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_Sortino.iloc[1]['Ticker']:<10} ({stats_Sortino.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_Sortino.iloc[2]['Ticker']:<10} ({stats_Sortino.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_Sortino.iloc[3]['Ticker']:<10} ({stats_Sortino.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_Sortino.iloc[4]['Ticker']:<10} ({stats_Sortino.iloc[4]['Days_Active']} días)")

# 11. GR1
print("\n   ► GR1:")
print(f"      1. {stats_GR1.iloc[0]['Ticker']:<10} ({stats_GR1.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_GR1.iloc[1]['Ticker']:<10} ({stats_GR1.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_GR1.iloc[2]['Ticker']:<10} ({stats_GR1.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_GR1.iloc[3]['Ticker']:<10} ({stats_GR1.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_GR1.iloc[4]['Ticker']:<10} ({stats_GR1.iloc[4]['Days_Active']} días)")

# 12. GR5
print("\n   ► GR5:")
print(f"      1. {stats_GR5.iloc[0]['Ticker']:<10} ({stats_GR5.iloc[0]['Days_Active']} días)")
print(f"      2. {stats_GR5.iloc[1]['Ticker']:<10} ({stats_GR5.iloc[1]['Days_Active']} días)")
print(f"      3. {stats_GR5.iloc[2]['Ticker']:<10} ({stats_GR5.iloc[2]['Days_Active']} días)")
print(f"      4. {stats_GR5.iloc[3]['Ticker']:<10} ({stats_GR5.iloc[3]['Days_Active']} días)")
print(f"      5. {stats_GR5.iloc[4]['Ticker']:<10} ({stats_GR5.iloc[4]['Days_Active']} días)")

print("\nTasa de supervivencia de los 5 ETFs que más tiempo han permanecido en cartera:")

# 1. SHARPE RATIO
print("\n   ► Sharpe Ratio:")
print(f"      1. {stats_SR.iloc[0]['Ticker']:<10} ({stats_SR.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_SR.iloc[1]['Ticker']:<10} ({stats_SR.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_SR.iloc[2]['Ticker']:<10} ({stats_SR.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_SR.iloc[3]['Ticker']:<10} ({stats_SR.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_SR.iloc[4]['Ticker']:<10} ({stats_SR.iloc[4]['Survival_Rate']:.2%})")

# 2. SHARPE CORRELATION
print("\n   ► Sharpe Correlation:")
print(f"      1. {stats_SR_corr.iloc[0]['Ticker']:<10} ({stats_SR_corr.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_SR_corr.iloc[1]['Ticker']:<10} ({stats_SR_corr.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_SR_corr.iloc[2]['Ticker']:<10} ({stats_SR_corr.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_SR_corr.iloc[3]['Ticker']:<10} ({stats_SR_corr.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_SR_corr.iloc[4]['Ticker']:<10} ({stats_SR_corr.iloc[4]['Survival_Rate']:.2%})")

# 3. MARGINAL SHARPE RATIO
print("\n   ► Marginal Sharpe Ratio (MSR):")
print(f"      1. {stats_MSR.iloc[0]['Ticker']:<10} ({stats_MSR.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_MSR.iloc[1]['Ticker']:<10} ({stats_MSR.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_MSR.iloc[2]['Ticker']:<10} ({stats_MSR.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_MSR.iloc[3]['Ticker']:<10} ({stats_MSR.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_MSR.iloc[4]['Ticker']:<10} ({stats_MSR.iloc[4]['Survival_Rate']:.2%})")

# 4. VAR RATIO 1%
print("\n   ► VaR Ratio 1%:")
print(f"      1. {stats_VaRR1.iloc[0]['Ticker']:<10} ({stats_VaRR1.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_VaRR1.iloc[1]['Ticker']:<10} ({stats_VaRR1.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_VaRR1.iloc[2]['Ticker']:<10} ({stats_VaRR1.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_VaRR1.iloc[3]['Ticker']:<10} ({stats_VaRR1.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_VaRR1.iloc[4]['Ticker']:<10} ({stats_VaRR1.iloc[4]['Survival_Rate']:.2%})")

# 5. VAR RATIO 5%
print("\n   ► VaR Ratio 5%:")
print(f"      1. {stats_VaRR5.iloc[0]['Ticker']:<10} ({stats_VaRR5.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_VaRR5.iloc[1]['Ticker']:<10} ({stats_VaRR5.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_VaRR5.iloc[2]['Ticker']:<10} ({stats_VaRR5.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_VaRR5.iloc[3]['Ticker']:<10} ({stats_VaRR5.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_VaRR5.iloc[4]['Ticker']:<10} ({stats_VaRR5.iloc[4]['Survival_Rate']:.2%})")

# 6. VAR RATIO 10%
print("\n   ► VaR Ratio 10%:")
print(f"      1. {stats_VaRR10.iloc[0]['Ticker']:<10} ({stats_VaRR10.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_VaRR10.iloc[1]['Ticker']:<10} ({stats_VaRR10.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_VaRR10.iloc[2]['Ticker']:<10} ({stats_VaRR10.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_VaRR10.iloc[3]['Ticker']:<10} ({stats_VaRR10.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_VaRR10.iloc[4]['Ticker']:<10} ({stats_VaRR10.iloc[4]['Survival_Rate']:.2%})")

# 7. OMEGA
print("\n   ► Omega:")
print(f"      1. {stats_Omega.iloc[0]['Ticker']:<10} ({stats_Omega.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_Omega.iloc[1]['Ticker']:<10} ({stats_Omega.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_Omega.iloc[2]['Ticker']:<10} ({stats_Omega.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_Omega.iloc[3]['Ticker']:<10} ({stats_Omega.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_Omega.iloc[4]['Ticker']:<10} ({stats_Omega.iloc[4]['Survival_Rate']:.2%})")

# 8. UPR
print("\n   ► UPR:")
print(f"      1. {stats_UPR.iloc[0]['Ticker']:<10} ({stats_UPR.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_UPR.iloc[1]['Ticker']:<10} ({stats_UPR.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_UPR.iloc[2]['Ticker']:<10} ({stats_UPR.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_UPR.iloc[3]['Ticker']:<10} ({stats_UPR.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_UPR.iloc[4]['Ticker']:<10} ({stats_UPR.iloc[4]['Survival_Rate']:.2%})")

# 9. KAPPA3
print("\n   ► Kappa3:")
print(f"      1. {stats_Kappa3.iloc[0]['Ticker']:<10} ({stats_Kappa3.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_Kappa3.iloc[1]['Ticker']:<10} ({stats_Kappa3.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_Kappa3.iloc[2]['Ticker']:<10} ({stats_Kappa3.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_Kappa3.iloc[3]['Ticker']:<10} ({stats_Kappa3.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_Kappa3.iloc[4]['Ticker']:<10} ({stats_Kappa3.iloc[4]['Survival_Rate']:.2%})")

# 10. SORTINO
print("\n   ► Sortino:")
print(f"      1. {stats_Sortino.iloc[0]['Ticker']:<10} ({stats_Sortino.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_Sortino.iloc[1]['Ticker']:<10} ({stats_Sortino.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_Sortino.iloc[2]['Ticker']:<10} ({stats_Sortino.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_Sortino.iloc[3]['Ticker']:<10} ({stats_Sortino.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_Sortino.iloc[4]['Ticker']:<10} ({stats_Sortino.iloc[4]['Survival_Rate']:.2%})")

# 11. GR1
print("\n   ► GR1:")
print(f"      1. {stats_GR1.iloc[0]['Ticker']:<10} ({stats_GR1.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_GR1.iloc[1]['Ticker']:<10} ({stats_GR1.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_GR1.iloc[2]['Ticker']:<10} ({stats_GR1.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_GR1.iloc[3]['Ticker']:<10} ({stats_GR1.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_GR1.iloc[4]['Ticker']:<10} ({stats_GR1.iloc[4]['Survival_Rate']:.2%})")

# 12. GR5
print("\n   ► GR5:")
print(f"      1. {stats_GR5.iloc[0]['Ticker']:<10} ({stats_GR5.iloc[0]['Survival_Rate']:.2%})")
print(f"      2. {stats_GR5.iloc[1]['Ticker']:<10} ({stats_GR5.iloc[1]['Survival_Rate']:.2%})")
print(f"      3. {stats_GR5.iloc[2]['Ticker']:<10} ({stats_GR5.iloc[2]['Survival_Rate']:.2%})")
print(f"      4. {stats_GR5.iloc[3]['Ticker']:<10} ({stats_GR5.iloc[3]['Survival_Rate']:.2%})")
print(f"      5. {stats_GR5.iloc[4]['Ticker']:<10} ({stats_GR5.iloc[4]['Survival_Rate']:.2%})")

print("\nTop 5 ETFs con mejor posición media en el ranking:")

# 1. SHARPE RATIO
print("\n   ► Sharpe Ratio:")
print(f"      1. {stats_SR.iloc[0]['Ticker']:<10} (Posición media: {stats_SR.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_SR.iloc[1]['Ticker']:<10} (Posición media: {stats_SR.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_SR.iloc[2]['Ticker']:<10} (Posición media: {stats_SR.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_SR.iloc[3]['Ticker']:<10} (Posición media: {stats_SR.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_SR.iloc[4]['Ticker']:<10} (Posición media: {stats_SR.iloc[4]['Average_Position']:.2f}º)")

# 2. SHARPE CORRELATION
print("\n   ► Sharpe Correlation:")
print(f"      1. {stats_SR_corr.iloc[0]['Ticker']:<10} (Posición media: {stats_SR_corr.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_SR_corr.iloc[1]['Ticker']:<10} (Posición media: {stats_SR_corr.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_SR_corr.iloc[2]['Ticker']:<10} (Posición media: {stats_SR_corr.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_SR_corr.iloc[3]['Ticker']:<10} (Posición media: {stats_SR_corr.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_SR_corr.iloc[4]['Ticker']:<10} (Posición media: {stats_SR_corr.iloc[4]['Average_Position']:.2f}º)")

# 3. MARGINAL SHARPE RATIO
print("\n   ► Marginal Sharpe Ratio (MSR):")
print(f"      1. {stats_MSR.iloc[0]['Ticker']:<10} (Posición media: {stats_MSR.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_MSR.iloc[1]['Ticker']:<10} (Posición media: {stats_MSR.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_MSR.iloc[2]['Ticker']:<10} (Posición media: {stats_MSR.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_MSR.iloc[3]['Ticker']:<10} (Posición media: {stats_MSR.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_MSR.iloc[4]['Ticker']:<10} (Posición media: {stats_MSR.iloc[4]['Average_Position']:.2f}º)")

# 4. VAR RATIO 1%
print("\n   ► VaR Ratio 1%:")
print(f"      1. {stats_VaRR1.iloc[0]['Ticker']:<10} (Posición media: {stats_VaRR1.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_VaRR1.iloc[1]['Ticker']:<10} (Posición media: {stats_VaRR1.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_VaRR1.iloc[2]['Ticker']:<10} (Posición media: {stats_VaRR1.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_VaRR1.iloc[3]['Ticker']:<10} (Posición media: {stats_VaRR1.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_VaRR1.iloc[4]['Ticker']:<10} (Posición media: {stats_VaRR1.iloc[4]['Average_Position']:.2f}º)")

# 5. VAR RATIO 5%
print("\n   ► VaR Ratio 5%:")
print(f"      1. {stats_VaRR5.iloc[0]['Ticker']:<10} (Posición media: {stats_VaRR5.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_VaRR5.iloc[1]['Ticker']:<10} (Posición media: {stats_VaRR5.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_VaRR5.iloc[2]['Ticker']:<10} (Posición media: {stats_VaRR5.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_VaRR5.iloc[3]['Ticker']:<10} (Posición media: {stats_VaRR5.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_VaRR5.iloc[4]['Ticker']:<10} (Posición media: {stats_VaRR5.iloc[4]['Average_Position']:.2f}º)")

# 6. VAR RATIO 10%
print("\n   ► VaR Ratio 10%:")
print(f"      1. {stats_VaRR10.iloc[0]['Ticker']:<10} (Posición media: {stats_VaRR10.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_VaRR10.iloc[1]['Ticker']:<10} (Posición media: {stats_VaRR10.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_VaRR10.iloc[2]['Ticker']:<10} (Posición media: {stats_VaRR10.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_VaRR10.iloc[3]['Ticker']:<10} (Posición media: {stats_VaRR10.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_VaRR10.iloc[4]['Ticker']:<10} (Posición media: {stats_VaRR10.iloc[4]['Average_Position']:.2f}º)")

# 7. OMEGA
print("\n   ► Omega:")
print(f"      1. {stats_Omega.iloc[0]['Ticker']:<10} (Posición media: {stats_Omega.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_Omega.iloc[1]['Ticker']:<10} (Posición media: {stats_Omega.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_Omega.iloc[2]['Ticker']:<10} (Posición media: {stats_Omega.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_Omega.iloc[3]['Ticker']:<10} (Posición media: {stats_Omega.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_Omega.iloc[4]['Ticker']:<10} (Posición media: {stats_Omega.iloc[4]['Average_Position']:.2f}º)")

# 8. UPR
print("\n   ► UPR:")
print(f"      1. {stats_UPR.iloc[0]['Ticker']:<10} (Posición media: {stats_UPR.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_UPR.iloc[1]['Ticker']:<10} (Posición media: {stats_UPR.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_UPR.iloc[2]['Ticker']:<10} (Posición media: {stats_UPR.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_UPR.iloc[3]['Ticker']:<10} (Posición media: {stats_UPR.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_UPR.iloc[4]['Ticker']:<10} (Posición media: {stats_UPR.iloc[4]['Average_Position']:.2f}º)")

# 9. KAPPA3
print("\n   ► Kappa3:")
print(f"      1. {stats_Kappa3.iloc[0]['Ticker']:<10} (Posición media: {stats_Kappa3.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_Kappa3.iloc[1]['Ticker']:<10} (Posición media: {stats_Kappa3.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_Kappa3.iloc[2]['Ticker']:<10} (Posición media: {stats_Kappa3.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_Kappa3.iloc[3]['Ticker']:<10} (Posición media: {stats_Kappa3.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_Kappa3.iloc[4]['Ticker']:<10} (Posición media: {stats_Kappa3.iloc[4]['Average_Position']:.2f}º)")

# 10. SORTINO
print("\n   ► Sortino:")
print(f"      1. {stats_Sortino.iloc[0]['Ticker']:<10} (Posición media: {stats_Sortino.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_Sortino.iloc[1]['Ticker']:<10} (Posición media: {stats_Sortino.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_Sortino.iloc[2]['Ticker']:<10} (Posición media: {stats_Sortino.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_Sortino.iloc[3]['Ticker']:<10} (Posición media: {stats_Sortino.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_Sortino.iloc[4]['Ticker']:<10} (Posición media: {stats_Sortino.iloc[4]['Average_Position']:.2f}º)")

# 11. GR1
print("\n   ► GR1:")
print(f"      1. {stats_GR1.iloc[0]['Ticker']:<10} (Posición media: {stats_GR1.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_GR1.iloc[1]['Ticker']:<10} (Posición media: {stats_GR1.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_GR1.iloc[2]['Ticker']:<10} (Posición media: {stats_GR1.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_GR1.iloc[3]['Ticker']:<10} (Posición media: {stats_GR1.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_GR1.iloc[4]['Ticker']:<10} (Posición media: {stats_GR1.iloc[4]['Average_Position']:.2f}º)")

# 12. GR5
print("\n   ► GR5:")
print(f"      1. {stats_GR5.iloc[0]['Ticker']:<10} (Posición media: {stats_GR5.iloc[0]['Average_Position']:.2f}º)")
print(f"      2. {stats_GR5.iloc[1]['Ticker']:<10} (Posición media: {stats_GR5.iloc[1]['Average_Position']:.2f}º)")
print(f"      3. {stats_GR5.iloc[2]['Ticker']:<10} (Posición media: {stats_GR5.iloc[2]['Average_Position']:.2f}º)")
print(f"      4. {stats_GR5.iloc[3]['Ticker']:<10} (Posición media: {stats_GR5.iloc[3]['Average_Position']:.2f}º)")
print(f"      5. {stats_GR5.iloc[4]['Ticker']:<10} (Posición media: {stats_GR5.iloc[4]['Average_Position']:.2f}º)")

# PARTE 4: CREAMOS NUEVO EXCEL
# CONVERTIMOS IDS A TICKERS
ranking_SR = ranking_SR_oos.map(id_to_ticker.get)
ranking_SR_corr = ranking_SR_corr_oos.map(id_to_ticker.get)
ranking_MSR = ranking_MSR_oos.map(id_to_ticker.get)
ranking_VaRR1 = ranking_VaRR1_oos.map(id_to_ticker.get)
ranking_VaRR5 = ranking_VaRR5_oos.map(id_to_ticker.get)
ranking_VaRR10 = ranking_VaRR10_oos.map(id_to_ticker.get)
ranking_Omega = ranking_Omega_oos.map(id_to_ticker.get)
ranking_UPR = ranking_UPR_oos.map(id_to_ticker.get)
ranking_Kappa3 = ranking_Kappa3_oos.map(id_to_ticker.get)
ranking_Sortino = ranking_Sortino_oos.map(id_to_ticker.get)
ranking_GR1 = ranking_GR1_oos.map(id_to_ticker.get)
ranking_GR5 = ranking_GR5_oos.map(id_to_ticker.get)

# PARTE 5: COSTES DE TRANSACCIÓN (DEMIGUEL 2009)
c = 0.0020

wealth_rolling_SR_netos = pd.Series(transaction_costs(R_oos, ids_rolling_SR, c=c), index=T)
wealth_rolling_SR_corr_netos = pd.Series(transaction_costs(R_oos, ids_rolling_SR_corr, c=c), index=T)
wealth_rolling_MSR_netos = pd.Series(transaction_costs(R_oos, ids_rolling_MSR, c=c), index=T)
wealth_rolling_VaRR1_netos = pd.Series(transaction_costs(R_oos, ids_rolling_VaRR1, c=c), index=T)
wealth_rolling_VaRR5_netos = pd.Series(transaction_costs(R_oos, ids_rolling_VaRR5, c=c), index=T)
wealth_rolling_VaRR10_netos = pd.Series(transaction_costs(R_oos, ids_rolling_VaRR10, c=c), index=T)
wealth_rolling_Omega_netos = pd.Series(transaction_costs(R_oos, ids_rolling_Omega, c=c), index=T)
wealth_rolling_UPR_netos = pd.Series(transaction_costs(R_oos, ids_rolling_UPR, c=c), index=T)
wealth_rolling_Kappa3_netos = pd.Series(transaction_costs(R_oos, ids_rolling_Kappa3, c=c), index=T)
wealth_rolling_Sortino_netos = pd.Series(transaction_costs(R_oos, ids_rolling_Sortino, c=c), index=T)
wealth_rolling_GR1_netos = pd.Series(transaction_costs(R_oos, ids_rolling_GR1, c=c), index=T)
wealth_rolling_GR5_netos = pd.Series(transaction_costs(R_oos, ids_rolling_GR5, c=c), index=T)

# GRÁFICO COMPARATIVO (PANEL 2X5): ROLLING SIN COSTES, CON COSTES Y BUY & HOLD
plot_cumulative_returns({
    'SR': (wealth_rolling_SR, wealth_buyhold_SR, wealth_rolling_SR_netos, 'blue'),
    'SR Correlation': (wealth_rolling_SR_corr, wealth_buyhold_SR_corr, wealth_rolling_SR_corr_netos, 'purple'),
    'MSR': (wealth_rolling_MSR, wealth_buyhold_MSR, wealth_rolling_MSR_netos, 'indigo'),
    'VaRR1': (wealth_rolling_VaRR1, wealth_buyhold_VaRR1, wealth_rolling_VaRR1_netos, 'red'),
    'VaRR5': (wealth_rolling_VaRR5, wealth_buyhold_VaRR5, wealth_rolling_VaRR5_netos, 'green'),
    'VaRR10': (wealth_rolling_VaRR10, wealth_buyhold_VaRR10, wealth_rolling_VaRR10_netos, 'orange'),
    'Omega': (wealth_rolling_Omega, wealth_buyhold_Omega, wealth_rolling_Omega_netos, 'blue'),
    'UPR': (wealth_rolling_UPR, wealth_buyhold_UPR, wealth_rolling_UPR_netos, 'green'),
    'Kappa3': (wealth_rolling_Kappa3, wealth_buyhold_Kappa3, wealth_rolling_Kappa3_netos, 'red'),
    'Sortino': (wealth_rolling_Sortino, wealth_buyhold_Sortino, wealth_rolling_Sortino_netos, 'orange'),
    'GR1': (wealth_rolling_GR1, wealth_buyhold_GR1, wealth_rolling_GR1_netos, 'brown'),
    'GR5': (wealth_rolling_GR5, wealth_buyhold_GR5, wealth_rolling_GR5_netos, 'green'),
})

# RESULTADOS FINALES
print("\nResultados: Riqueza Acumulada")

# BUY & HOLD
print("\n   ► BUY & HOLD (sin costes):")
print(f"      Sharpe Ratio : €{wealth_buyhold_SR.iloc[-1]:.4f}")
print(f"      Sharpe Corr  : €{wealth_buyhold_SR_corr.iloc[-1]:.4f}")
print(f"      MSR          : €{wealth_buyhold_MSR.iloc[-1]:.4f}")
print(f"      VaRR1        : €{wealth_buyhold_VaRR1.iloc[-1]:.4f}")
print(f"      VaRR5        : €{wealth_buyhold_VaRR5.iloc[-1]:.4f}")
print(f"      VaRR10       : €{wealth_buyhold_VaRR10.iloc[-1]:.4f}")
print(f"      Omega        : €{wealth_buyhold_Omega.iloc[-1]:.4f}")
print(f"      UPR          : €{wealth_buyhold_UPR.iloc[-1]:.4f}")
print(f"      Kappa3       : €{wealth_buyhold_Kappa3.iloc[-1]:.4f}")
print(f"      Sortino      : €{wealth_buyhold_Sortino.iloc[-1]:.4f}")
print(f"      GR1          : €{wealth_buyhold_GR1.iloc[-1]:.4f}")
print(f"      GR5          : €{wealth_buyhold_GR5.iloc[-1]:.4f}")

# ROLLING SIN COSTES
print("\n   ► ROLLING WINDOW (sin costes):")
print(f"      Sharpe Ratio : €{wealth_rolling_SR.iloc[-1]:.4f}")
print(f"      Sharpe Corr  : €{wealth_rolling_SR_corr.iloc[-1]:.4f}")
print(f"      MSR          : €{wealth_rolling_MSR.iloc[-1]:.4f}")
print(f"      VaRR1        : €{wealth_rolling_VaRR1.iloc[-1]:.4f}")
print(f"      VaRR5        : €{wealth_rolling_VaRR5.iloc[-1]:.4f}")
print(f"      VaRR10       : €{wealth_rolling_VaRR10.iloc[-1]:.4f}")
print(f"      Omega        : €{wealth_rolling_Omega.iloc[-1]:.4f}")
print(f"      UPR          : €{wealth_rolling_UPR.iloc[-1]:.4f}")
print(f"      Kappa3       : €{wealth_rolling_Kappa3.iloc[-1]:.4f}")
print(f"      Sortino      : €{wealth_rolling_Sortino.iloc[-1]:.4f}")
print(f"      GR1          : €{wealth_rolling_GR1.iloc[-1]:.4f}")
print(f"      GR5          : €{wealth_rolling_GR5.iloc[-1]:.4f}")

# ROLLING CON COSTES
print(f"\n   ► ROLLING WINDOW - con costes de transacción - ({c * 10000} bps):")
print(f"      Sharpe Ratio : €{wealth_rolling_SR_netos.iloc[-1]:.4f}")
print(f"      Sharpe Corr  : €{wealth_rolling_SR_corr_netos.iloc[-1]:.4f}")
print(f"      MSR          : €{wealth_rolling_MSR_netos.iloc[-1]:.4f}")
print(f"      VaRR1        : €{wealth_rolling_VaRR1_netos.iloc[-1]:.4f}")
print(f"      VaRR5        : €{wealth_rolling_VaRR5_netos.iloc[-1]:.4f}")
print(f"      VaRR10       : €{wealth_rolling_VaRR10_netos.iloc[-1]:.4f}")
print(f"      Omega        : €{wealth_rolling_Omega_netos.iloc[-1]:.4f}")
print(f"      UPR          : €{wealth_rolling_UPR_netos.iloc[-1]:.4f}")
print(f"      Kappa3       : €{wealth_rolling_Kappa3_netos.iloc[-1]:.4f}")
print(f"      Sortino      : €{wealth_rolling_Sortino_netos.iloc[-1]:.4f}")
print(f"      GR1          : €{wealth_rolling_GR1_netos.iloc[-1]:.4f}")
print(f"      GR5          : €{wealth_rolling_GR5_netos.iloc[-1]:.4f}")

plt.show(block=True)

print(f"\n Análisis completado. Archivo guardado: {OUTPUT_FILE}")

# VARIABLES GUARADAS PARA EL PROXIMO CÓDIGO
variables = {
    'K': K,
    'c': c,
    'T': T,
    'M': M,
    'df_rendimientos': df_rendimientos,
    'tickers': tickers,
    'ticker_to_id': ticker_to_id,
    'id_to_ticker': id_to_ticker,
    'oos_start_date': OOS_START_DATE,
    'rankings_ids': {
        'SR': ids_rolling_SR,
        'SR_corr': ids_rolling_SR_corr,
        'MSR': ids_rolling_MSR,
        'VaRR1': ids_rolling_VaRR1,
        'VaRR5': ids_rolling_VaRR5,
        'VaRR10': ids_rolling_VaRR10,
        'Omega': ids_rolling_Omega,
        'UPR': ids_rolling_UPR,
        'Kappa3': ids_rolling_Kappa3,
        'Sortino': ids_rolling_Sortino,
        'GR1': ids_rolling_GR1,
        'GR5': ids_rolling_GR5,
    },
    'rotation_mean': {
        'SR': float(df_turnover_activos['Nuevos_SR'].mean()),
        'SR_corr': float(df_turnover_activos['Nuevos_SR_corr'].mean()),
        'MSR': float(df_turnover_activos['Nuevos_MSR'].mean()),
        'VaRR1': float(df_turnover_activos['Nuevos_VaRR1'].mean()),
        'VaRR5': float(df_turnover_activos['Nuevos_VaRR5'].mean()),
        'VaRR10': float(df_turnover_activos['Nuevos_VaRR10'].mean()),
        'Omega': float(df_turnover_activos['Nuevos_Omega'].mean()),
        'UPR': float(df_turnover_activos['Nuevos_UPR'].mean()),
        'Kappa3': float(df_turnover_activos['Nuevos_Kappa3'].mean()),
        'Sortino': float(df_turnover_activos['Nuevos_Sortino'].mean()),
        'GR1': float(df_turnover_activos['Nuevos_GR1'].mean()),
        'GR5': float(df_turnover_activos['Nuevos_GR5'].mean()),
    }
}

joblib.dump(variables, VARIABLES_FILE)

# PARTE 4: CREAMOS SERIES TEMPORALES DE LOS RATIOS DE PERFORMANCE
# FECHA INICIO DE LA SERIE
fecha_inicio_IS = df_rendimientos.index[M].strftime('%Y-%m-%d') 

# GENERAMOS LAS SERIES
rankings_completa, series_completa = ROLLING_WINDOW_RANKINGS(df_rendimientos, oos_start_date=fecha_inicio_IS, K=K, ratios_list=ratios)

serie_SR_completa = series_completa['SR']
serie_SR_corr_completa = series_completa['SR_corr']
serie_MSR_completa = series_completa['MSR']
serie_VaRR1_completa = series_completa['VaRR1']
serie_VaRR5_completa = series_completa['VaRR5']
serie_VaRR10_completa = series_completa['VaRR10']
serie_Omega_completa = series_completa['Omega']
serie_UPR_completa = series_completa['UPR']
serie_Kappa3_completa = series_completa['Kappa3']
serie_Sortino_completa = series_completa['Sortino']
serie_GR1_completa = series_completa['GR1']
serie_GR5_completa = series_completa['GR5']

"""
# PASAMOS A EXCEL
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl', mode='w') as writer:
    
    # ID DE ETFS
    df_ID.to_excel(writer, sheet_name='IDs', index=False)
    
    # SERIES TEMPORALES OUT OF SAMPLE
    serie_SR_oos.to_excel(writer, sheet_name='Serie Sharpe OOS')
    serie_SR_corr_oos.to_excel(writer, sheet_name='Serie Sharpe Corr OOS')
    serie_MSR_oos.to_excel(writer, sheet_name='Serie Marginal Sharpe OOS')
    serie_VaRR1_oos.to_excel(writer, sheet_name='Serie VaRR1 OOS')
    serie_VaRR5_oos.to_excel(writer, sheet_name='Serie VaRR5 OOS')
    serie_VaRR10_oos.to_excel(writer, sheet_name='Serie VaRR10 OOS')
    serie_Omega_oos.to_excel(writer, sheet_name='Serie Omega OOS')
    serie_UPR_oos.to_excel(writer, sheet_name='Serie UPR OOS')
    serie_Kappa3_oos.to_excel(writer, sheet_name='Serie Kappa3 OOS')
    serie_Sortino_oos.to_excel(writer, sheet_name='Serie Sortino OOS')
    serie_GR1_oos.to_excel(writer, sheet_name='Serie GR1 OOS')
    serie_GR5_oos.to_excel(writer, sheet_name='Serie GR5 OOS')
    
    # SERIES TEMPORALES COMPLETAS
    serie_SR_completa.to_excel(writer, sheet_name='Serie Sharpe COMPLETA')
    serie_SR_corr_completa.to_excel(writer, sheet_name='Serie Sharpe Corr COMPLETA')
    serie_MSR_completa.to_excel(writer, sheet_name='Serie Marginal Sharpe COMPLETA')
    serie_VaRR1_completa.to_excel(writer, sheet_name='Serie VaRR1 COMPLETA')
    serie_VaRR5_completa.to_excel(writer, sheet_name='Serie VaRR5 COMPLETA')
    serie_VaRR10_completa.to_excel(writer, sheet_name='Serie VaRR10 COMPLETA')
    serie_Omega_completa.to_excel(writer, sheet_name='Serie Omega COMPLETA')
    serie_UPR_completa.to_excel(writer, sheet_name='Serie UPR COMPLETA')
    serie_Kappa3_completa.to_excel(writer, sheet_name='Serie Kappa3 COMPLETA')
    serie_Sortino_completa.to_excel(writer, sheet_name='Serie Sortino COMPLETA')
    serie_GR1_completa.to_excel(writer, sheet_name='Serie GR1 COMPLETA')
    serie_GR5_completa.to_excel(writer, sheet_name='Serie GR5 COMPLETA')
    
    # RANKINGS (TOP K) OUT OF SAMPLE
    ranking_SR.to_excel(writer, sheet_name='Ranking Sharpe OOS')
    ranking_SR_corr.to_excel(writer, sheet_name='Ranking Sharpe Corr OOS')
    ranking_MSR.to_excel(writer, sheet_name='Ranking Marginal Sharpe OOS')
    ranking_VaRR1.to_excel(writer, sheet_name='Ranking VaRR1 OOS')
    ranking_VaRR5.to_excel(writer, sheet_name='Ranking VaRR5 OOS')
    ranking_VaRR10.to_excel(writer, sheet_name='Ranking VaRR10 OOS')
    ranking_Omega.to_excel(writer, sheet_name='Ranking Omega OOS')
    ranking_UPR.to_excel(writer, sheet_name='Ranking UPR OOS')
    ranking_Kappa3.to_excel(writer, sheet_name='Ranking Kappa3 OOS')
    ranking_Sortino.to_excel(writer, sheet_name='Ranking Sortino OOS')
    ranking_GR1.to_excel(writer, sheet_name='Ranking GR1 OOS')
    ranking_GR5.to_excel(writer, sheet_name='Ranking GR5 OOS')
    
    # REBALANCEO DE ETFS
    df_turnover_activos.to_excel(writer, sheet_name='Rebalanceo diario')
    df_turnover_freq.to_excel(writer, sheet_name='Frecuencia rebalanceo')

    # ESTADÍSTICAS DE POSICIÓN DE ETFS
    stats_SR.to_excel(writer, sheet_name='Survival_SR', index=False)
    stats_SR_corr.to_excel(writer, sheet_name='Survival_SR_corr', index=False)
    stats_MSR.to_excel(writer, sheet_name='Survival_MSR', index=False)
    stats_VaRR1.to_excel(writer, sheet_name='Survival_VaRR1', index=False)
    stats_VaRR5.to_excel(writer, sheet_name='Survival_VaRR5', index=False)
    stats_VaRR10.to_excel(writer, sheet_name='Survival_VaRR10', index=False)
    stats_Omega.to_excel(writer, sheet_name='Survival_Omega', index=False)
    stats_UPR.to_excel(writer, sheet_name='Survival_UPR', index=False)
    stats_Kappa3.to_excel(writer, sheet_name='Survival_Kappa3', index=False)
    stats_Sortino.to_excel(writer, sheet_name='Survival_Sortino', index=False)
    stats_GR1.to_excel(writer, sheet_name='Survival_GR1', index=False)
    stats_GR5.to_excel(writer, sheet_name='Survival_GR5', index=False)
"""