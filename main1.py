"""
main1.py
PARTE I: ROLLING WINDOW RANKINGS (SELECCIÓN DINÁMICA DE CARTERAS)
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martínez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from screening import ROLLING_WINDOW_RANKINGS
from metrics import etf_stats, replacement_rate, transaction_costs, coincident_assets_median
from plots import plot_cumulative_returns, plot_turnover_frequency, plot_coincidence_heatmap, plot_survival_heatmap
from portfolios import W_EW

INPUT_FILE = "100_ETFs_Smart_Beta.xlsx"
OUTPUT_FILE = "100_ETFs_Screening.xlsx"
VARIABLES_FILE = "screening_artifacts.joblib"

# ETFs EN CARTERA
K = 20

# VENTANA IN SAMPLE
M = 1000

# COSTES DE TRANSACCIÓN
c = 0.0020

# FAMILIAS SMART BETA
SMART_BETA_FAMILIES = {
    "Value": ["VTV", "IVE", "IWD", "IWN", "IJJ", "IJS", "IUSV", "VOE", "IWS", "IWX", "MGV", "RPV", "VIOV", "SCHV", "SPYV", "ILCV", "IMCV", "ISCV", "PWV", "RFV", "RZV", "EFV", "VOOV", "VBR"],
    "Growth": ["DNL", "VUG", "IWF", "IVW", "SCHG", "SPYG", "IUSG", "IJK", "IJT", "IWO", "IWP", "IWY", "MDYG", "VBK", "VONG", "VOOG", "VOT", "ILCG", "IMCG", "ISCG", "RFG", "RPG", "RZG", "EFG"],
    "Dividend": ["VIG", "VYM", "SCHD", "DVY", "SDY", "HDV", "DHS", "DLN", "DON", "DTD", "DTN", "FVD", "PFM", "DES", "DLS", "DEW", "DFE", "DFJ", "DIM", "DTH", "DOL", "DWM"],
    "Low Volatility": ["USMV", "SPLV", "EFAV", "EEMV", "ACWV", "EELV"],
    "Defensive": ["SPHQ", "PKW", "FTCS", "MOAT", "QDEF", "PDP", "PIZ"],
    "Fundamental": ["PRF", "PXF", "PXH", "PRFZ", "EZM", "EES"],
    "ESG": ["TILT", "FEM", "FEMS", "SUSA", "DSI"],
    "Alternative": ["IYY", "FDN", "FPX", "KIE", "RSP", "DEF"]
}

ticker_to_macrofamily = {ticker: family for family, tickers in SMART_BETA_FAMILIES.items() for ticker in tickers}

# EXTRAEMOS PRECIOS (P_T): (M + T X 100)
df_precios = pd.read_excel(INPUT_FILE, sheet_name='Precios_Historicos', index_col=0, header=1)
df_precios = df_precios.iloc[1:]
df_precios.index = pd.to_datetime(df_precios.index).normalize()
df_precios = df_precios.astype(float)

# RENDIMIENTOS ARITMÉTICOS (P_T / P_T-1 - 1): (M + T - 1 X 100)
df_rendimientos = df_precios.pct_change().dropna(how='all')
OOS_START_DATE = df_rendimientos.index[M]

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
dias_totales = len(df_rendimientos)
dias_is = len(df_rendimientos[df_rendimientos.index < OOS_START_DATE])
dias_oos = len(df_rendimientos[df_rendimientos.index >= OOS_START_DATE])
pct_nulos = 100.0 * df_rendimientos.isna().sum().sum() / df_rendimientos.size

print("DATOS:")
print(" Configuración de la Cartera ")
print(f"  • Universo de ETFs (N):     {len(tickers)}")
print(f"  • Objetivo en cartera (K):  {K}")
print(f"  • Costes de transacc. (c):  {c}")

print("\n Horizonte Temporal ")
print(f"  • Fecha inicial global:     {df_rendimientos.index.min().date()}")
print(f"  • Fecha inicio OOS:         {OOS_START_DATE.date()}")
print(f"  • Fecha final global:       {df_rendimientos.index.max().date()}")

print("\n Muestra ")
print(f"  • Días totales:             {dias_totales}")
print(f"  • Días In-Sample (IS):      {dias_is}")
print(f"  • Días Out-of-Sample (OOS): {dias_oos}")
print(f"  • Tamaño de Ventana IS (M): {M} días")
print(f"  • NaNs:                     {pct_nulos:.4f} %")

# PARTE 1.I: SIMULACIÓN DE RANKINGS - ROLLING WINDOW (M = 1000 DÍAS)

ratios = ['SR', 'SR_corr', 'MSR', 'VaRR1', 'VaRR5', 'VaRR10', 'Omega', 'UPR', 'Kappa3', 'Sortino', 'GR1', 'GR5']
rankings_oos, series_oos = ROLLING_WINDOW_RANKINGS(df_rendimientos, OOS_START_DATE, K, ratios, M=M)

# RANKINGS OOS (FECHAS DESDE OOS_START_DATE HASTA EL FINAL)
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

# PERIODO OUT OF SAMPLE (T)
T = df_rendimientos.index[df_rendimientos.index >= OOS_START_DATE]
t = ranking_SR_oos.index.get_indexer(T)

# LOS ACTIVOS SELECCIONADOS PARA GENERAR EL RENDIMIENTO DE HOY (t) SON LOS SELECCIONADOS AYER (t-1)
t_prev = np.maximum(0, t - 1)

# MATRICES DE IDS SELECCIONADOS: (T X K)
ids_rolling_SR = (ranking_SR_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_SR_corr = (ranking_SR_corr_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_MSR = (ranking_MSR_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_VaRR5 = (ranking_VaRR5_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_VaRR1 = (ranking_VaRR1_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_VaRR10 = (ranking_VaRR10_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_Omega = (ranking_Omega_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_UPR = (ranking_UPR_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_Kappa3 = (ranking_Kappa3_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_Sortino = (ranking_Sortino_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_GR1 = (ranking_GR1_oos.iloc[t_prev].values - 1).astype(int)
ids_rolling_GR5 = (ranking_GR5_oos.iloc[t_prev].values - 1).astype(int)

print("\nPARTE 1.1. - ANÁLISIS DE COINCIDENCIA DE ACTIVOS (MEDIANA HISTÓRICA %)")

ranking_ids = {
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
    'GR5': ids_rolling_GR5
}

ratios_list = list(ranking_ids.keys())
coincidence_matrix = pd.DataFrame(index=ratios_list, columns=ratios_list, dtype=float)

for r1 in ratios_list:
    for r2 in ratios_list:
        if r1 == r2:
            coincidence_matrix.loc[r1, r2] = 100.0
        else:
            med_coinc = coincident_assets_median(ranking_ids[r1], ranking_ids[r2], K)
            coincidence_matrix.loc[r1, r2] = np.round(med_coinc, 2)

print("\nMatriz Mediana de Coincidencia de Activos (%):")
print()
print(coincidence_matrix.to_string())

plot_coincidence_heatmap(
    coincidence_matrix, 
    save_path="heatmap_coincidencia_fase1.pdf"
)

# PARTE 1.II: ANÁLISIS DEL REBALANCEO

print("\nPARTE 1.2. - ANÁLISIS DEL REBALANCEO DE LOS ACTIVOS")

# CALCULAMOS EL NÚMERO DE ETFS QUE ENTRAN CADA DÍA A LA CARTERA
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
print(f"   • Sharpe               : {df_turnover_activos['Nuevos_SR'].mean():.4f} de {K} ETFs por día")
print(f"   • Sharpe Correlation   : {df_turnover_activos['Nuevos_SR_corr'].mean():.4f} de {K} ETFs por día")
print(f"   • MSR                  : {df_turnover_activos['Nuevos_MSR'].mean():.4f} de {K} ETFs por día")
print(f"   • VaRR1                : {df_turnover_activos['Nuevos_VaRR1'].mean():.4f} de {K} ETFs por día")
print(f"   • VaRR5                : {df_turnover_activos['Nuevos_VaRR5'].mean():.4f} de {K} ETFs por día")
print(f"   • VaRR10               : {df_turnover_activos['Nuevos_VaRR10'].mean():.4f} de {K} ETFs por día")
print(f"   • Omega                : {df_turnover_activos['Nuevos_Omega'].mean():.4f} de {K} ETFs por día")
print(f"   • UPR                  : {df_turnover_activos['Nuevos_UPR'].mean():.4f} de {K} ETFs por día")
print(f"   • Kappa3               : {df_turnover_activos['Nuevos_Kappa3'].mean():.4f} de {K} ETFs por día")
print(f"   • Sortino              : {df_turnover_activos['Nuevos_Sortino'].mean():.4f} de {K} ETFs por día")
print(f"   • GR1                  : {df_turnover_activos['Nuevos_GR1'].mean():.4f} de {K} ETFs por día")
print(f"   • GR5                  : {df_turnover_activos['Nuevos_GR5'].mean():.4f} de {K} ETFs por día")

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
    'GR5': freq_GR5
})

# FRECUENCIAS DE REBALANCEO DIARIO
print("\nFrecuencia de rebalanceo (Número de ETFs nuevos en cartera):")

for ratio in ratios:
    print(f"\n   ► {ratio}:")
    for i in range(5):
        print(f"      • {i} ETFs nuevos: {df_turnover_freq[ratio].get(i, 0)} veces")

# GRÁFICO DE FRECUENCIAS DE REBALANCEO
plot_turnover_frequency({
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
        'GR5': freq_GR5},
    save_path="turnover_histograms.pdf"
)

# PARTE 1.III: ANÁLISIS DE LOS COSTES DE TRANSACCIÓN

print("\nPARTE 1.3. - IMPACTO DE LOS COSTES DE TRANSACCIÓN EN LA RIQUEZA")

# CARTERA EQUIPONDERADA
w = W_EW(K)

# A) SIMULACIÓN DE CARTERA DE SELECCIÓN DINÁMICA (ROLLING)
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

# RIQUEZA ACUMULADA BRUTA DE LA CARTERA ROLLING
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

# B) CARTERA REBALANCING B&H (EQUIPONDERADA CON REBALANCEO DIARIO)

# 1. K MEJORES ETFS EN T=0
ids_buyhold_SR_pos = ids_rolling_SR[0]
ids_buyhold_SR_corr_pos = ids_rolling_SR_corr[0]
ids_buyhold_MSR_pos = ids_rolling_MSR[0]
ids_buyhold_VaRR1_pos = ids_rolling_VaRR1[0]
ids_buyhold_VaRR5_pos = ids_rolling_VaRR5[0]
ids_buyhold_VaRR10_pos = ids_rolling_VaRR10[0]
ids_buyhold_Omega_pos = ids_rolling_Omega[0]
ids_buyhold_UPR_pos = ids_rolling_UPR[0]
ids_buyhold_Kappa3_pos = ids_rolling_Kappa3[0]
ids_buyhold_Sortino_pos = ids_rolling_Sortino[0]
ids_buyhold_GR1_pos = ids_rolling_GR1[0]
ids_buyhold_GR5_pos = ids_rolling_GR5[0]

# 2. RENDIMIENTO DIARIO EQUIPONDERADO DE LA CARTERA FIJA (REBALANCEO FORZADO)
R_port_reb_SR = pd.Series(R_oos[:, ids_buyhold_SR_pos] @ w, index=T)
R_port_reb_SR_corr = pd.Series(R_oos[:, ids_buyhold_SR_corr_pos] @ w, index=T)
R_port_reb_MSR = pd.Series(R_oos[:, ids_buyhold_MSR_pos] @ w, index=T)
R_port_reb_VaRR1 = pd.Series(R_oos[:, ids_buyhold_VaRR1_pos] @ w, index=T)
R_port_reb_VaRR5 = pd.Series(R_oos[:, ids_buyhold_VaRR5_pos] @ w, index=T)
R_port_reb_VaRR10 = pd.Series(R_oos[:, ids_buyhold_VaRR10_pos] @ w, index=T)
R_port_reb_Omega = pd.Series(R_oos[:, ids_buyhold_Omega_pos] @ w, index=T)
R_port_reb_UPR = pd.Series(R_oos[:, ids_buyhold_UPR_pos] @ w, index=T)
R_port_reb_Kappa3 = pd.Series(R_oos[:, ids_buyhold_Kappa3_pos] @ w, index=T)
R_port_reb_Sortino = pd.Series(R_oos[:, ids_buyhold_Sortino_pos] @ w, index=T)
R_port_reb_GR1 = pd.Series(R_oos[:, ids_buyhold_GR1_pos] @ w, index=T)
R_port_reb_GR5 = pd.Series(R_oos[:, ids_buyhold_GR5_pos] @ w, index=T)

# 3. RIQUEZA ACUMULADA BRUTA DEL BENCHMARK REBALANCEADO
wealth_buyhold_SR = W0 * (1 + R_port_reb_SR).cumprod()
wealth_buyhold_SR_corr = W0 * (1 + R_port_reb_SR_corr).cumprod()
wealth_buyhold_MSR = W0 * (1 + R_port_reb_MSR).cumprod()
wealth_buyhold_VaRR1 = W0 * (1 + R_port_reb_VaRR1).cumprod()
wealth_buyhold_VaRR5 = W0 * (1 + R_port_reb_VaRR5).cumprod()
wealth_buyhold_VaRR10 = W0 * (1 + R_port_reb_VaRR10).cumprod()
wealth_buyhold_Omega = W0 * (1 + R_port_reb_Omega).cumprod()
wealth_buyhold_UPR = W0 * (1 + R_port_reb_UPR).cumprod()
wealth_buyhold_Kappa3 = W0 * (1 + R_port_reb_Kappa3).cumprod()
wealth_buyhold_Sortino = W0 * (1 + R_port_reb_Sortino).cumprod()
wealth_buyhold_GR1 = W0 * (1 + R_port_reb_GR1).cumprod()
wealth_buyhold_GR5 = W0 * (1 + R_port_reb_GR5).cumprod()

# C) COSTES DE TRANSACCIÓN (c) Y RIQUEZA NETA

# MATRICES CONSTANTES PARA EL BENCHMARK (T X K)
ids_bh_matrix_SR = np.tile(ids_buyhold_SR_pos, (len(T), 1))
ids_bh_matrix_SR_corr = np.tile(ids_buyhold_SR_corr_pos, (len(T), 1))
ids_bh_matrix_MSR = np.tile(ids_buyhold_MSR_pos, (len(T), 1))
ids_bh_matrix_VaRR1 = np.tile(ids_buyhold_VaRR1_pos, (len(T), 1))
ids_bh_matrix_VaRR5 = np.tile(ids_buyhold_VaRR5_pos, (len(T), 1))
ids_bh_matrix_VaRR10 = np.tile(ids_buyhold_VaRR10_pos, (len(T), 1))
ids_bh_matrix_Omega = np.tile(ids_buyhold_Omega_pos, (len(T), 1))
ids_bh_matrix_UPR = np.tile(ids_buyhold_UPR_pos, (len(T), 1))
ids_bh_matrix_Kappa3 = np.tile(ids_buyhold_Kappa3_pos, (len(T), 1))
ids_bh_matrix_Sortino = np.tile(ids_buyhold_Sortino_pos, (len(T), 1))
ids_bh_matrix_GR1 = np.tile(ids_buyhold_GR1_pos, (len(T), 1))
ids_bh_matrix_GR5 = np.tile(ids_buyhold_GR5_pos, (len(T), 1))

# RIQUEZA NETA - ESTRATEGIA DINÁMICA (SUSTITUCIÓN + DERIVA)
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

# RIQUEZA NETA - BENCHMARK ESTÁTICO (SÓLO DERIVA)
wealth_buyhold_SR_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_SR, c=c), index=T)
wealth_buyhold_SR_corr_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_SR_corr, c=c), index=T)
wealth_buyhold_MSR_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_MSR, c=c), index=T)
wealth_buyhold_VaRR1_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_VaRR1, c=c), index=T)
wealth_buyhold_VaRR5_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_VaRR5, c=c), index=T)
wealth_buyhold_VaRR10_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_VaRR10, c=c), index=T)
wealth_buyhold_Omega_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_Omega, c=c), index=T)
wealth_buyhold_UPR_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_UPR, c=c), index=T)
wealth_buyhold_Kappa3_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_Kappa3, c=c), index=T)
wealth_buyhold_Sortino_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_Sortino, c=c), index=T)
wealth_buyhold_GR1_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_GR1, c=c), index=T)
wealth_buyhold_GR5_netos = pd.Series(transaction_costs(R_oos, ids_bh_matrix_GR5, c=c), index=T)

# D) GRÁFICOS Y RESULTADOS
diccionario_resultados = {
    'SR':      (wealth_rolling_SR, wealth_rolling_SR_netos, wealth_buyhold_SR, wealth_buyhold_SR_netos, 'blue'),
    'SR_corr': (wealth_rolling_SR_corr, wealth_rolling_SR_corr_netos, wealth_buyhold_SR_corr, wealth_buyhold_SR_corr_netos, 'purple'),
    'MSR':     (wealth_rolling_MSR, wealth_rolling_MSR_netos, wealth_buyhold_MSR, wealth_buyhold_MSR_netos, 'indigo'),
    'VaRR1':   (wealth_rolling_VaRR1, wealth_rolling_VaRR1_netos, wealth_buyhold_VaRR1, wealth_buyhold_VaRR1_netos, 'red'),
    'VaRR5':   (wealth_rolling_VaRR5, wealth_rolling_VaRR5_netos, wealth_buyhold_VaRR5, wealth_buyhold_VaRR5_netos, 'green'),
    'VaRR10':  (wealth_rolling_VaRR10, wealth_rolling_VaRR10_netos, wealth_buyhold_VaRR10, wealth_buyhold_VaRR10_netos, 'orange'),
    'Omega':   (wealth_rolling_Omega, wealth_rolling_Omega_netos, wealth_buyhold_Omega, wealth_buyhold_Omega_netos, 'blue'),
    'UPR':     (wealth_rolling_UPR, wealth_rolling_UPR_netos, wealth_buyhold_UPR, wealth_buyhold_UPR_netos, 'green'),
    'Kappa3':  (wealth_rolling_Kappa3, wealth_rolling_Kappa3_netos, wealth_buyhold_Kappa3, wealth_buyhold_Kappa3_netos, 'red'),
    'Sortino': (wealth_rolling_Sortino, wealth_rolling_Sortino_netos, wealth_buyhold_Sortino, wealth_buyhold_Sortino_netos, 'orange'),
    'GR1':     (wealth_rolling_GR1, wealth_rolling_GR1_netos, wealth_buyhold_GR1, wealth_buyhold_GR1_netos, 'brown'),
    'GR5':     (wealth_rolling_GR5, wealth_rolling_GR5_netos, wealth_buyhold_GR5, wealth_buyhold_GR5_netos, 'green')
}

plot_cumulative_returns(diccionario_resultados)

# RESULTADOS DE LA RIQUEZA ACUMULADA
print("\nRESULTADOS: RIQUEZA ACUMULADA (BAJO COSTES DE TRANSACCIÓN): SELECCIÓN ACTIVA VS PASIVA")

# a) PASIVA SIN COSTES
print("\n   ► EQUIPONDERADA CON REBALANCEO DIARIO (SELECCIÓN PASIVA) [sin costes]:")
print(f"     Sharpe Ratio : €{wealth_buyhold_SR.iloc[-1]:.4f}")
print(f"     Sharpe Corr  : €{wealth_buyhold_SR_corr.iloc[-1]:.4f}")
print(f"     MSR          : €{wealth_buyhold_MSR.iloc[-1]:.4f}")
print(f"     VaRR1        : €{wealth_buyhold_VaRR1.iloc[-1]:.4f}")
print(f"     VaRR5        : €{wealth_buyhold_VaRR5.iloc[-1]:.4f}")
print(f"     VaRR10       : €{wealth_buyhold_VaRR10.iloc[-1]:.4f}")
print(f"     Omega        : €{wealth_buyhold_Omega.iloc[-1]:.4f}")
print(f"     UPR          : €{wealth_buyhold_UPR.iloc[-1]:.4f}")
print(f"     Kappa3       : €{wealth_buyhold_Kappa3.iloc[-1]:.4f}")
print(f"     Sortino      : €{wealth_buyhold_Sortino.iloc[-1]:.4f}")
print(f"     GR1          : €{wealth_buyhold_GR1.iloc[-1]:.4f}")
print(f"     GR5          : €{wealth_buyhold_GR5.iloc[-1]:.4f}")

# b) ACTIVA SIN COSTES
print("\n   ► SELECCIÓN DINÁMICA [sin costes]:")
print(f"     Sharpe Ratio : €{wealth_rolling_SR.iloc[-1]:.4f}")
print(f"     Sharpe Corr  : €{wealth_rolling_SR_corr.iloc[-1]:.4f}")
print(f"     MSR          : €{wealth_rolling_MSR.iloc[-1]:.4f}")
print(f"     VaRR1        : €{wealth_rolling_VaRR1.iloc[-1]:.4f}")
print(f"     VaRR5        : €{wealth_rolling_VaRR5.iloc[-1]:.4f}")
print(f"     VaRR10       : €{wealth_rolling_VaRR10.iloc[-1]:.4f}")
print(f"     Omega        : €{wealth_rolling_Omega.iloc[-1]:.4f}")
print(f"     UPR          : €{wealth_rolling_UPR.iloc[-1]:.4f}")
print(f"     Kappa3       : €{wealth_rolling_Kappa3.iloc[-1]:.4f}")
print(f"     Sortino      : €{wealth_rolling_Sortino.iloc[-1]:.4f}")
print(f"     GR1          : €{wealth_rolling_GR1.iloc[-1]:.4f}")
print(f"     GR5          : €{wealth_rolling_GR5.iloc[-1]:.4f}")

# c) PASIVA CON COSTES
print(f"\n   ► EQUIPONDERADA CON REBALANCEO DIARIO (SELECCIÓN PASIVA) [c = {c * 10000} bps]:")
print(f"     Sharpe Ratio : €{wealth_buyhold_SR_netos.iloc[-1]:.4f}")
print(f"     Sharpe Corr  : €{wealth_buyhold_SR_corr_netos.iloc[-1]:.4f}")
print(f"     MSR          : €{wealth_buyhold_MSR_netos.iloc[-1]:.4f}")
print(f"     VaRR1        : €{wealth_buyhold_VaRR1_netos.iloc[-1]:.4f}")
print(f"     VaRR5        : €{wealth_buyhold_VaRR5_netos.iloc[-1]:.4f}")
print(f"     VaRR10       : €{wealth_buyhold_VaRR10_netos.iloc[-1]:.4f}")
print(f"     Omega        : €{wealth_buyhold_Omega_netos.iloc[-1]:.4f}")
print(f"     UPR          : €{wealth_buyhold_UPR_netos.iloc[-1]:.4f}")
print(f"     Kappa3       : €{wealth_buyhold_Kappa3_netos.iloc[-1]:.4f}")
print(f"     Sortino      : €{wealth_buyhold_Sortino_netos.iloc[-1]:.4f}")
print(f"     GR1          : €{wealth_buyhold_GR1_netos.iloc[-1]:.4f}")
print(f"     GR5          : €{wealth_buyhold_GR5_netos.iloc[-1]:.4f}")

# d) ACTIVA CON COSTES
print(f"\n   ► SELECCIÓN DINÁMICA [c = {c * 10000} bps]:")
print(f"     Sharpe Ratio : €{wealth_rolling_SR_netos.iloc[-1]:.4f}")
print(f"     Sharpe Corr  : €{wealth_rolling_SR_corr_netos.iloc[-1]:.4f}")
print(f"     MSR          : €{wealth_rolling_MSR_netos.iloc[-1]:.4f}")
print(f"     VaRR1        : €{wealth_rolling_VaRR1_netos.iloc[-1]:.4f}")
print(f"     VaRR5        : €{wealth_rolling_VaRR5_netos.iloc[-1]:.4f}")
print(f"     VaRR10       : €{wealth_rolling_VaRR10_netos.iloc[-1]:.4f}")
print(f"     Omega        : €{wealth_rolling_Omega_netos.iloc[-1]:.4f}")
print(f"     UPR          : €{wealth_rolling_UPR_netos.iloc[-1]:.4f}")
print(f"     Kappa3       : €{wealth_rolling_Kappa3_netos.iloc[-1]:.4f}")
print(f"     Sortino      : €{wealth_rolling_Sortino_netos.iloc[-1]:.4f}")
print(f"     GR1          : €{wealth_rolling_GR1_netos.iloc[-1]:.4f}")
print(f"     GR5          : €{wealth_rolling_GR5_netos.iloc[-1]:.4f}")

plt.show(block=True)

# PARTE 1.IV: ESTADÍSTICAS DE SELECCIÓN

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

ratio_names = {
    'SR': 'Sharpe Ratio',
    'SR_corr': 'Sharpe Correlation',
    'MSR': 'Marginal Sharpe Ratio (MSR)',
    'VaRR1': 'VaR Ratio 1%',
    'VaRR5': 'VaR Ratio 5%',
    'VaRR10': 'VaR Ratio 10%',
    'Omega': 'Omega',
    'UPR': 'UPR',
    'Kappa3': 'Kappa3',
    'Sortino': 'Sortino',
    'GR1': 'GR1',
    'GR5': 'GR5'
}

all_stats_dict = {
    'SR': stats_SR,
    'SR_corr': stats_SR_corr,
    'MSR': stats_MSR,
    'VaRR1': stats_VaRR1,
    'VaRR5': stats_VaRR5,
    'VaRR10': stats_VaRR10,
    'Omega': stats_Omega,
    'UPR': stats_UPR,
    'Kappa3': stats_Kappa3,
    'Sortino': stats_Sortino,
    'GR1': stats_GR1,
    'GR5': stats_GR5
}

print("\nPARTE 1.4. - ESTADÍSTICAS DE SELECCIÓN DE ACTIVOS (TOP 5)")

# 1. DÍAS EN CARTERA
print("\nTop 5 ETFs que más tiempo han permanecido en cartera:")
for key, name in ratio_names.items():
    print(f"\n   ► {name}:")
    df = all_stats_dict[key]
    for i in range(5):
        print(f"      {i+1}. {df.iloc[i]['Ticker']:<10} ({df.iloc[i]['Days_Active']} días)")

# 2. TASA DE SUPERVIVENCIA
print("\nTasa de supervivencia de los 5 ETFs que más tiempo han permanecido en cartera:")
for key, name in ratio_names.items():
    print(f"\n   ► {name}:")
    df = all_stats_dict[key]
    for i in range(5):
        print(f"      {i+1}. {df.iloc[i]['Ticker']:<10} ({df.iloc[i]['Survival_Rate']:.2%})")

# 3. POSICIÓN MEDIANA
print("\nTop 5 ETFs con mejor posición mediana en el ranking:")
for key, name in ratio_names.items():
    print(f"\n   ► {name}:")
    df = all_stats_dict[key]
    for i in range(5):
        print(f"      {i+1}. {df.iloc[i]['Ticker']:<10} (Posición mediana: {df.iloc[i]['Median_Position']:.2f}º)")

# 4. FAMILIAS SMART BETA
print("\nFamilias de los Top 5 ETFs que más tiempo han permanecido en cartera:")
for key, name in ratio_names.items():
    print(f"\n   ► {name}:")
    df = all_stats_dict[key]
    for i in range(5):
        ticker = df.iloc[i]['Ticker']
        family = ticker_to_macrofamily.get(ticker, "Desconocida")
        print(f"      {i+1}. {ticker:<10} ({family})")

# GRÁFICO CONJUNTO
plot_survival_heatmap(all_stats_dict, ticker_to_macrofamily)

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

# ANEXO C.1: ANÁLISIS DE ROBUSTEZ TAMAÑO DE LAS CARTERAS (K=10 vs K=20)
print("\n ANEXO C.1. - ANÁLISIS DE SENSIBILIDAD (K=10 vs K=20)")

# 1. PARÁMETROS Y RANKINGS EXCLUSIVOS PARA TOP 10
K_rob = 10
w_rob = W_EW(K_rob)

rankings_oos_K10, _ = ROLLING_WINDOW_RANKINGS(df_rendimientos, OOS_START_DATE, K_rob, ratios, M=M)

# 2. MATRICES DE IDs SELECCIONADOS (K=10)
ids_rolling_SR_K10      = (rankings_oos_K10['SR'].iloc[t_prev].values - 1).astype(int)
ids_rolling_SR_corr_K10 = (rankings_oos_K10['SR_corr'].iloc[t_prev].values - 1).astype(int)
ids_rolling_MSR_K10     = (rankings_oos_K10['MSR'].iloc[t_prev].values - 1).astype(int)
ids_rolling_VaRR1_K10   = (rankings_oos_K10['VaRR1'].iloc[t_prev].values - 1).astype(int)
ids_rolling_VaRR5_K10   = (rankings_oos_K10['VaRR5'].iloc[t_prev].values - 1).astype(int)
ids_rolling_VaRR10_K10  = (rankings_oos_K10['VaRR10'].iloc[t_prev].values - 1).astype(int)
ids_rolling_Omega_K10   = (rankings_oos_K10['Omega'].iloc[t_prev].values - 1).astype(int)
ids_rolling_UPR_K10     = (rankings_oos_K10['UPR'].iloc[t_prev].values - 1).astype(int)
ids_rolling_Kappa3_K10  = (rankings_oos_K10['Kappa3'].iloc[t_prev].values - 1).astype(int)
ids_rolling_Sortino_K10 = (rankings_oos_K10['Sortino'].iloc[t_prev].values - 1).astype(int)
ids_rolling_GR1_K10     = (rankings_oos_K10['GR1'].iloc[t_prev].values - 1).astype(int)
ids_rolling_GR5_K10     = (rankings_oos_K10['GR5'].iloc[t_prev].values - 1).astype(int)

# 3. RENDIMIENTOS Y RIQUEZA BRUTA (K=10)
wealth_rolling_SR_K10      = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_SR_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_SR_corr_K10 = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_SR_corr_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_MSR_K10     = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_MSR_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_VaRR1_K10   = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_VaRR1_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_VaRR5_K10   = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_VaRR5_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_VaRR10_K10  = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_VaRR10_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_Omega_K10   = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_Omega_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_UPR_K10     = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_UPR_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_Kappa3_K10  = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_Kappa3_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_Sortino_K10 = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_Sortino_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_GR1_K10     = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_GR1_K10] @ w_rob, index=T)).cumprod()
wealth_rolling_GR5_K10     = W0 * (1 + pd.Series(R_oos[t_oos, ids_rolling_GR5_K10] @ w_rob, index=T)).cumprod()

# 4. RIQUEZA NETA (K=10) CON COSTES DE TRANSACCIÓN
wealth_rolling_SR_netos_K10      = pd.Series(transaction_costs(R_oos, ids_rolling_SR_K10, c=c), index=T)
wealth_rolling_SR_corr_netos_K10 = pd.Series(transaction_costs(R_oos, ids_rolling_SR_corr_K10, c=c), index=T)
wealth_rolling_MSR_netos_K10     = pd.Series(transaction_costs(R_oos, ids_rolling_MSR_K10, c=c), index=T)
wealth_rolling_VaRR1_netos_K10   = pd.Series(transaction_costs(R_oos, ids_rolling_VaRR1_K10, c=c), index=T)
wealth_rolling_VaRR5_netos_K10   = pd.Series(transaction_costs(R_oos, ids_rolling_VaRR5_K10, c=c), index=T)
wealth_rolling_VaRR10_netos_K10  = pd.Series(transaction_costs(R_oos, ids_rolling_VaRR10_K10, c=c), index=T)
wealth_rolling_Omega_netos_K10   = pd.Series(transaction_costs(R_oos, ids_rolling_Omega_K10, c=c), index=T)
wealth_rolling_UPR_netos_K10     = pd.Series(transaction_costs(R_oos, ids_rolling_UPR_K10, c=c), index=T)
wealth_rolling_Kappa3_netos_K10  = pd.Series(transaction_costs(R_oos, ids_rolling_Kappa3_K10, c=c), index=T)
wealth_rolling_Sortino_netos_K10 = pd.Series(transaction_costs(R_oos, ids_rolling_Sortino_K10, c=c), index=T)
wealth_rolling_GR1_netos_K10     = pd.Series(transaction_costs(R_oos, ids_rolling_GR1_K10, c=c), index=T)
wealth_rolling_GR5_netos_K10     = pd.Series(transaction_costs(R_oos, ids_rolling_GR5_K10, c=c), index=T)

# 5. DICCIONARIO CARTERAS ROLLING (K=10 VS K=20)
diccionario_K10_vs_K20 = {
    'SR':      (wealth_rolling_SR_K10,      wealth_rolling_SR_netos_K10,      wealth_rolling_SR,      wealth_rolling_SR_netos,      'blue'),
    'SR_corr': (wealth_rolling_SR_corr_K10, wealth_rolling_SR_corr_netos_K10, wealth_rolling_SR_corr, wealth_rolling_SR_corr_netos,       'purple'),
    'MSR':     (wealth_rolling_MSR_K10,     wealth_rolling_MSR_netos_K10,     wealth_rolling_MSR,     wealth_rolling_MSR_netos,     'indigo'),
    'VaRR1':   (wealth_rolling_VaRR1_K10,   wealth_rolling_VaRR1_netos_K10,   wealth_rolling_VaRR1,   wealth_rolling_VaRR1_netos,   'red'),
    'VaRR5':   (wealth_rolling_VaRR5_K10,   wealth_rolling_VaRR5_netos_K10,   wealth_rolling_VaRR5,   wealth_rolling_VaRR5_netos,   'green'),
    'VaRR10':  (wealth_rolling_VaRR10_K10,  wealth_rolling_VaRR10_netos_K10,  wealth_rolling_VaRR10,  wealth_rolling_VaRR10_netos,  'orange'),
    'Omega':   (wealth_rolling_Omega_K10,   wealth_rolling_Omega_netos_K10,   wealth_rolling_Omega,   wealth_rolling_Omega_netos,   'blue'),
    'UPR':     (wealth_rolling_UPR_K10,     wealth_rolling_UPR_netos_K10,     wealth_rolling_UPR,     wealth_rolling_UPR_netos,     'green'),
    'Kappa3':  (wealth_rolling_Kappa3_K10,  wealth_rolling_Kappa3_netos_K10,  wealth_rolling_Kappa3,  wealth_rolling_Kappa3_netos,  'red'),
    'Sortino': (wealth_rolling_Sortino_K10, wealth_rolling_Sortino_netos_K10, wealth_rolling_Sortino, wealth_rolling_Sortino_netos, 'orange'),
    'GR1':     (wealth_rolling_GR1_K10,     wealth_rolling_GR1_netos_K10,     wealth_rolling_GR1,     wealth_rolling_GR1_netos,     'brown'),
    'GR5':     (wealth_rolling_GR5_K10,     wealth_rolling_GR5_netos_K10,     wealth_rolling_GR5,     wealth_rolling_GR5_netos,     'green')
}

# 6. GRÁFICO DE SPREADS
plot_cumulative_returns(diccionario_K10_vs_K20)
print("   Gráfico comparativo de rendimientos acumulados (K=10 vs K=20) generado.")

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
    'rankings_ids': ranking_ids,
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

# 7. EXPORTACIÓN A EXCEL
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl', mode='w') as writer:
    df_ID.to_excel(writer, sheet_name='IDs', index=False)
    
    for ratio in ratios:
        series_oos[ratio].to_excel(writer, sheet_name=f'Serie {ratio} OOS')
        rankings_oos[ratio].map(id_to_ticker.get).to_excel(writer, sheet_name=f'Ranking {ratio} OOS')
        all_stats_dict[ratio].to_excel(writer, sheet_name=f'Survival_{ratio}', index=False)
        
    df_turnover_activos.to_excel(writer, sheet_name='Rebalanceo diario')
    df_turnover_freq.to_excel(writer, sheet_name='Frecuencia rebalanceo')
    
print("\n PARTE I COMPLETADA")
print(f"   • Cálculos exportados en : {OUTPUT_FILE}")
print(f"   • Artefactos del modelo guardados en: {VARIABLES_FILE}")