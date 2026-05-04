"""
main2.py
PARTE II: GESTIÓN DE CARTERAS, RIESGO Y PERFORMANCE
Smart Beta ETF Universe - TFM
Autor: Alejandro Martínez
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from screening import ROLLING_WINDOW_RANKINGS
from metrics import risk_contribution, allocation_model_with_costs, MHI, maxDrawDown
from plots import plot_allocation_spreads, plot_selection_spreads, plot_mhi, plot_robustness_analysis, plot_dendrogram, plot_matrix_heatmap, plot_dendrogram_heatmap, plot_single_allocation_spread, plot_single_selection_spread
from portfolios import W_EW, W_VT, W_RRT, W_GMV, W_ERC, W_MVS, W_HRP, W_HERC
from hrp import HRP


import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from hrp import correlDist, getQuasiDiag

OUTPUT_FILE = "100_ETFs_Allocation.xlsx"
VARIABLES_FILE = "screening_artifacts.joblib"
VERTICAL_DATE = pd.to_datetime("2024-12-19")

# RANKINGS DE SELECCIÓN (Top K por ratio) Y PERFILES DE PESOS
ratios = ['SR', 'SR_corr', 'MSR', 'VaRR1', 'VaRR5', 'VaRR10', 'Omega', 'UPR', 'Kappa3', 'Sortino', 'GR1', 'GR5']
perfiles = ['EW', 'VT', 'RRT', 'GMV', 'ERC', 'MVS', 'HRP', 'HERC']

# RECUPERAMOS VARIABLES
variables = joblib.load(VARIABLES_FILE)

K = variables['K']
c = variables['c']
T = variables['T']
M = variables['M']
tickers = variables['tickers']
id_to_ticker = variables['id_to_ticker']
df_rendimientos = variables['df_rendimientos']
OOS_START_DATE = variables['oos_start_date']
rendimientos = df_rendimientos.values
rankings = variables['rankings_ids']
rotation_mean = variables['rotation_mean']

# ÍNDICES DE LOS DÍAS OUT-OF-SAMPLE (T)
t_indices = df_rendimientos.index.searchsorted(T)

# DATOS
dias_totales = len(df_rendimientos)
dias_is = len(df_rendimientos[df_rendimientos.index < OOS_START_DATE])
dias_oos = len(T)
pct_nulos = 100.0 * df_rendimientos.isna().sum().sum() / df_rendimientos.size

print(" DATOS")

print("\n Configuración de la Cartera ")
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

# MATRICES DE RANKINGS (T x K), UNA POR CADA RATIO
# COMPARAMOS N ESTRATEGIAS DISTINTAS DE INVERSIÓN (12 RATIOS x 8 PERFILES)
print("\nCalculando rendimientos out-of-sample para todas las carteras.")
print(f"\nRatios de selección ({len(ratios)}): {', '.join(ratios)}")
print(f"\nPerfiles de pesos ({len(perfiles)}): {', '.join(perfiles)}")

# 8 PERFILES DE PESOS (w)
perfiles_de_pesos = [('EW', W_EW), ('VT', W_VT), ('RRT', W_RRT), ('GMV', W_GMV), ('ERC', W_ERC), ('MVS', W_MVS), ('HRP', W_HRP), ('HERC', W_HERC)]
rendimientos_oos = {}
riqueza_oos = {}

# SERIES DEL ÍNDICE DE HERFINDAHL: 12 x 8 x (T x 1)
mhi_history = {ratio: {perfil: np.zeros(len(T)) for perfil in perfiles} for ratio in ratios}

# BUCLE EXTERIOR - FIJAMOS LOS IDs GANADORES (T x K)
for ratio in ratios:
    print(f"\n Ratio de Selección: {ratio}")
    rendimientos_oos[ratio] = {}
    riqueza_oos[ratio] = {}
    ids_matrix = rankings[ratio]  # EXTRAEMOS LA MATRIZ DEL RATIO ACTUAL (T x K)
    
    # BUCLE INTERIOR - ITERAMOS LA MATRIZ DE GANADORES POR LOS 8 PERFILES
    for perfil, model in perfiles_de_pesos:
        print(f"  -> Optimizando pesos según: {perfil}")
        
        # CADA DÍA CALCULAMOS EL MHI Y LO GUARDAMOS EN LA SERIE
        def save_mhi(t, w_k, ratio_key=ratio, perfil_key=perfil):
            mhi_history[ratio_key][perfil_key][t] = MHI(w_k)    # len(ratios) x 8 x (T x 1)

        r_neto, riq_neta = allocation_model_with_costs(T, t_indices, M, K, rendimientos, ids_matrix, model, perfil, c=c, save_mhi=save_mhi)
        
        rendimientos_oos[ratio][perfil] = r_neto   # 12 x 8 x (T x 1)
        riqueza_oos[ratio][perfil] = riq_neta      # 12 x 8 x (T x 1)

    
# CREAMOS MATRICES DE RENDIMIENTOS OOS POR RATIO: 12 x (T x 8)
rendimientos_oos_por_ratio = {}
for ratio in ratios:
    rendimientos_oos_por_ratio[ratio] = pd.DataFrame(
        {perfil: rendimientos_oos[ratio][perfil] for perfil in perfiles}, index=T)

# CONTRIBUCIONES AL RIESGO EN EL DÍA DE LA VERTICAL, ELEGIMOS CUALQUIER RATIO
ratio_vertical = "GR1"

# FIJAMOS A UN DÍA CONCRETO Y UN RATIO CONCRETO
i = T.get_loc(VERTICAL_DATE)
    
# ELEGIMOS EL RATIO DE SELECCIÓN PARA LA VERTICAL
ids_ratio = rankings[ratio_vertical]               # (T x K)
ids_vertical = ids_ratio[i]                        # (K x 1)
    
# VENTANA IN-SAMPLE  (M DÍAS HACIA ATRÁS)
t_i = t_indices[i]
R_IS = rendimientos[t_i - M : t_i, ids_vertical]   # (M x K)
mu = R_IS.mean(axis=0)                             # (K x 1)
Sigma = np.cov(R_IS, rowvar=False, ddof=1)         # (K x K)
Sigma = (Sigma + Sigma.T) / 2.0

# ETFs SELECCIONADOS EN LA VERTICAL (PASAMOS DE IDs A TICKERS)
etfs_vertical = [id_to_ticker[ids + 1] for ids in ids_vertical]

# CORRELATION MATRIX
corr_matrix = pd.DataFrame(R_IS, columns=etfs_vertical).corr()

# ALGORITMO HRP
w_HRP, links_vertical_HRP, Sigma_quasidiag_HRP = HRP(Sigma, labels=etfs_vertical)

# ALGORITMO HERC (Reconstrucción topológica para los gráficos estáticos)
dist = correlDist(corr_matrix)
links_vertical_HERC = sch.linkage(ssd.squareform(dist, checks=False), method='ward')

# Obtenemos el orden cuasi-diagonal leyendo el linkage de Ward
sort_ix_HERC = getQuasiDiag(links_vertical_HERC)
etfs_ordenados_HERC = corr_matrix.columns[sort_ix_HERC].tolist()

# CORRELACIONES CUASIDIAGONALIZADAS
corr_quasidiag_HRP = corr_matrix.loc[Sigma_quasidiag_HRP.index, Sigma_quasidiag_HRP.columns]
corr_quasidiag_HERC = corr_matrix.loc[etfs_ordenados_HERC, etfs_ordenados_HERC]

# DENDROGRAMAS
plot_dendrogram(
    links_vertical_HRP, labels=etfs_vertical, 
    title="HRP - Dendrograma de ETFs (Single Linkage)", save_path="dendrograma_HRP.pdf"
)

plot_dendrogram(
    links_vertical_HERC, labels=etfs_vertical, 
    title="HERC - Dendrograma de ETFs (Ward Linkage)", save_path="dendrograma_HERC.pdf"
)

# HEATMAPS DE CORRELACIONES
plot_matrix_heatmap(
    corr_matrix, 
    title=f'Correlaciones ({ratio_vertical}) - Original',
    save_path="heatmap_corr_original.pdf"
)

plot_matrix_heatmap(
    corr_quasidiag_HRP, 
    title=f'Correlaciones Cuasidiagonalizada ({ratio_vertical}) - HRP',
    save_path="heatmap_corr_HRP.pdf"
)

plot_matrix_heatmap(
    corr_quasidiag_HERC, 
    title=f'Correlaciones Cuasidiagonalizada ({ratio_vertical}) - HERC',
    save_path="heatmap_corr_HERC.pdf"
)

# CLUSTERMAPS (DENDROGRAMA + HEATMAP)
plot_dendrogram_heatmap(
    corr_matrix, link=links_vertical_HRP, 
    title=f"HRP - Estructura Jerárquica y Correlación (Single Linkage) | {ratio_vertical}", save_path="clustermap_HRP.pdf"
)

plot_dendrogram_heatmap(
    corr_matrix, link=links_vertical_HERC, 
    title=f"HERC - Estructura Jerárquica y Correlación (Ward Linkage) | {ratio_vertical}", save_path="clustermap_HERC.pdf"
)

print(f"\nTop {K} ETFs del día {VERTICAL_DATE.date()} según el ratio {ratio_vertical}: {etfs_vertical}")
    
# PESOS (w)
w_EW = W_EW(K)                                      # (K x 1)
w_VT = W_VT(Sigma, K)                               # (K x 1)
w_RRT = W_RRT(mu, Sigma, K)                         # (K x 1)
w_GMV = w_GMV = W_GMV(R_IS, K, print_delta=True)    # (K x 1)
w_ERC = W_ERC(Sigma, K)                             # (K x 1)
w_MVS = W_MVS(R_IS, mu, Sigma, K)                   # (K x 1)
w_HRP = w_HRP                                       # (K x 1)
w_HERC = W_HERC(R_IS, K)                            # (K x 1)

# MEDIAS (mu_p) IN SAMPLE
mu_EW = float(w_EW @ mu)                    # (K x 1) @ (K x 1) -> escalar
mu_VT = float(w_VT @ mu)                    # (K x 1) @ (K x 1) -> escalar
mu_RRT = float(w_RRT @ mu)                  # (K x 1) @ (K x 1) -> escalar
mu_GMV = float(w_GMV @ mu)                  # (K x 1) @ (K x 1) -> escalar
mu_ERC = float(w_ERC @ mu)                  # (K x 1) @ (K x 1) -> escalar
mu_MVS = float(w_MVS @ mu)                  # (K x 1) @ (K x 1) -> escalar
mu_HRP = float(w_HRP @ mu)                  # (K x 1) @ (K x 1) -> escalar
mu_HERC = float(w_HERC @ mu)                # (K x 1) @ (K x 1) -> escalar

# VARIANZAS (var_p) IN SAMPLE
var_EW = float(w_EW.T @ Sigma @ w_EW)       # (1 x K) @ (K x K) @ (K x 1) -> escalar
var_VT = float(w_VT.T @ Sigma @ w_VT)       # (1 x K) @ (K x K) @ (K x 1) -> escalar
var_RRT = float(w_RRT.T @ Sigma @ w_RRT)    # (1 x K) @ (K x K) @ (K x 1) -> escalar
var_GMV = float(w_GMV.T @ Sigma @ w_GMV)    # (1 x K) @ (K x K) @ (K x 1) -> escalar
var_ERC = float(w_ERC.T @ Sigma @ w_ERC)    # (1 x K) @ (K x K) @ (K x 1) -> escalar
var_MVS = float(w_MVS.T @ Sigma @ w_MVS)    # (1 x K) @ (K x K) @ (K x 1) -> escalar
var_HRP = float(w_HRP.T @ Sigma @ w_HRP)    # (1 x K) @ (K x K) @ (K x 1) -> escalar
var_HERC = float(w_HERC.T @ Sigma @ w_HERC) # (1 x K) @ (K x K) @ (K x 1) -> escalar

# RATIO DE SHARPE (SR) IN SAMPLE
SR_EW = (mu_EW / np.sqrt(var_EW)) * np.sqrt(252)
SR_VT = (mu_VT / np.sqrt(var_VT)) * np.sqrt(252)
SR_RRT = (mu_RRT / np.sqrt(var_RRT)) * np.sqrt(252)
SR_GMV = (mu_GMV / np.sqrt(var_GMV)) * np.sqrt(252)
SR_ERC = (mu_ERC / np.sqrt(var_ERC)) * np.sqrt(252)
SR_MVS = (mu_MVS / np.sqrt(var_MVS)) * np.sqrt(252)
SR_HRP = (mu_HRP / np.sqrt(var_HRP)) * np.sqrt(252)
SR_HERC = (mu_HERC / np.sqrt(var_HERC)) * np.sqrt(252)
    
# RISK CONTRIBUTION (RC) ABSOLUTA Y RELATIVA (%)
RC_abs_EW, RC_rel_EW = risk_contribution(w_EW, Sigma)   
RC_abs_VT, RC_rel_VT = risk_contribution(w_VT, Sigma)
RC_abs_RRT, RC_rel_RRT = risk_contribution(w_RRT, Sigma)
RC_abs_GMV, RC_rel_GMV = risk_contribution(w_GMV, Sigma)
RC_abs_ERC, RC_rel_ERC = risk_contribution(w_ERC, Sigma)
RC_abs_MVS, RC_rel_MVS = risk_contribution(w_MVS, Sigma)
RC_abs_HRP, RC_rel_HRP = risk_contribution(w_HRP, Sigma)
RC_abs_HERC, RC_rel_HERC = risk_contribution(w_HERC, Sigma)

# 1. EQUIPONDERADA (EW)
print("\n1. EQUALLYWEIGHTED (EW)")
print(f"   ► Media                   : {mu_EW:.8f}")
print(f"   ► Varianza                : {var_EW:.10f}")
print(f"   ► Volatilidad             : {np.sqrt(var_EW):.4f}")
print(f"   ► Ratio de Sharpe         : {SR_EW:.4f}")
print(f"   ► Pesos                   : {np.round(w_EW * 100, 2)}%")
print(f"   ► RC Absoluta             : {np.round(RC_abs_EW, 4)}")
print(f"   ► RC Relativa (%)         : {np.round(RC_rel_EW * 100, 2)}%")

# 2. VOLATILITY TIMING (VT)
print("\n2. VOLATILITY TIMING (VT)")
print(f"   ► Media                   : {mu_VT:.8f}")
print(f"   ► Varianza                : {var_VT:.10f}")
print(f"   ► Volatilidad             : {np.sqrt(var_VT):.4f}")
print(f"   ► Ratio de Sharpe         : {SR_VT:.4f}")
print(f"   ► Pesos                   : {np.round(w_VT * 100, 2)}%")
print(f"   ► RC Absoluta             : {np.round(RC_abs_VT, 4)}")
print(f"   ► RC Relativa (%)         : {np.round(RC_rel_VT * 100, 2)}%")

# 3. REWARD-TO-RISK TIMING (RRT)
print("\n3. REWARD-TO-RISK TIMING (RRT)")
print(f"   ► Media                   : {mu_RRT:.8f}")
print(f"   ► Varianza                : {var_RRT:.10f}")
print(f"   ► Volatilidad             : {np.sqrt(var_RRT):.4f}")
print(f"   ► Ratio de Sharpe         : {SR_RRT:.4f}")
print(f"   ► Pesos                   : {np.round(w_RRT * 100, 2)}%")
print(f"   ► RC Absoluta             : {np.round(RC_abs_RRT, 4)}")
print(f"   ► RC Relativa (%)         : {np.round(RC_rel_RRT * 100, 2)}%")
    
# 4. MINIMA VARIANZA GLOBAL (GMV)
print("\n4. GLOBAL MINIMUM VARIANCE (GMV)")
print(f"   ► Media                   : {mu_GMV:.8f}")
print(f"   ► Varianza                : {var_GMV:.10f}")
print(f"   ► Volatilidad             : {np.sqrt(var_GMV):.4f}")
print(f"   ► Ratio de Sharpe         : {SR_GMV:.4f}")
print(f"   ► Pesos                   : {np.round(w_GMV * 100, 2)}%")
print(f"   ► RC Absoluta             : {np.round(RC_abs_GMV, 4)}")
print(f"   ► RC Relativa (%)         : {np.round(RC_rel_GMV * 100, 2)}%")

# 5. EQUAL RISK CONTRIBUTION (ERC)
print("\n5. EQUAL RISK CONTRIBUTION (ERC)")
print(f"   ► Media                   : {mu_ERC:.8f}")
print(f"   ► Varianza                : {var_ERC:.10f}")
print(f"   ► Volatilidad             : {np.sqrt(var_ERC):.4f}")
print(f"   ► Ratio de Sharpe         : {SR_ERC:.4f}")
print(f"   ► Pesos                   : {np.round(w_ERC * 100, 2)}%")
print(f"   ► RC Absoluta             : {np.round(RC_abs_ERC, 4)}")
print(f"   ► RC Relativa (%)         : {np.round(RC_rel_ERC * 100, 2)}%")

# 6. MEAN-VARIANCE-SKEWNESS (MVS)
print("\n6. MEAN-VARIANCE-SKEWNESS (MVS)")
print(f"   ► Media                   : {mu_MVS:.8f}")
print(f"   ► Varianza                : {var_MVS:.10f}")
print(f"   ► Volatilidad             : {np.sqrt(var_MVS):.4f}")
print(f"   ► Ratio de Sharpe         : {SR_MVS:.4f}")
print(f"   ► Pesos                   : {np.round(w_MVS * 100, 2)}%")
print(f"   ► RC Absoluta             : {np.round(RC_abs_MVS, 4)}")
print(f"   ► RC Relativa (%)         : {np.round(RC_rel_MVS * 100, 2)}%")

# 7. HIERARCHICAL RISK PARITY (HRP)
print("\n7. HIERARCHICAL RISK PARITY (HRP)")
print(f"   ► Media                   : {mu_HRP:.8f}")
print(f"   ► Varianza                : {var_HRP:.10f}")
print(f"   ► Volatilidad             : {np.sqrt(var_HRP):.4f}")
print(f"   ► Ratio de Sharpe         : {SR_HRP:.4f}")
print(f"   ► Pesos                   : {np.round(w_HRP * 100, 2)}%")
print(f"   ► RC Absoluta             : {np.round(RC_abs_HRP, 4)}")
print(f"   ► RC Relativa (%)         : {np.round(RC_rel_HRP * 100, 2)}%")

# 8. HIERARCHICAL EQUAL RISK CONTRIBUTION (HERC)
print("\n8. HIERARCHICAL EQUAL RISK CONTRIBUTION (HERC)")
print(f"   ► Media                   : {mu_HERC:.8f}")
print(f"   ► Varianza                : {var_HERC:.10f}")
print(f"   ► Volatilidad             : {np.sqrt(var_HERC):.4f}")
print(f"   ► Ratio de Sharpe         : {SR_HERC:.4f}")
print(f"   ► Pesos                   : {np.round(w_HERC * 100, 2)}%")
print(f"   ► RC Absoluta             : {np.round(RC_abs_HERC, 4)}")
print(f"   ► RC Relativa (%)         : {np.round(RC_rel_HERC * 100, 2)}%")

# CREAMOS MATRICES DE RIQUEZA OOS POR RATIO: 12 x (T x 8)
riqueza_oos_por_ratio = {}
for ratio in ratios:
    riqueza_oos_por_ratio[ratio] = pd.DataFrame(
        {perfil: riqueza_oos[ratio][perfil] for perfil in perfiles}, index=T)

# MATRIZ DE RENDIMIENTOS NETOS DIARIOS: (T x 12 * 8)
df_rendimientos_estrategias = pd.concat(rendimientos_oos_por_ratio, axis=1)
df_rendimientos_estrategias.columns = [f"{ratio}_{perfil}" for ratio, perfil in df_rendimientos_estrategias.columns]

# ANUALIZACIÓN DE RENDIMIENTOS Y VOLATILIDADES (ASUMIENDO 252 DÍAS POR AÑO)
annual_factor = 252

# TABLA DE RESULTADOS (12 * 8 x 4)
resultados_oos = pd.DataFrame({
    'Rentabilidad Anualizada': df_rendimientos_estrategias.mean() * annual_factor,
    'Volatilidad Anualizada': df_rendimientos_estrategias.std() * np.sqrt(annual_factor),
    'Sharpe Ratio Anualizado': (df_rendimientos_estrategias.mean() / df_rendimientos_estrategias.std()) * np.sqrt(annual_factor),
    'Asimetría (Skewness)': df_rendimientos_estrategias.skew(),
}).sort_values(by='Sharpe Ratio Anualizado', ascending=False)

# GRÁFICOS DE SPREADS

# I) QUÉ RATIO ELIGE MEJOR?
selection_spreads_vs_sr_benchmark = plot_selection_spreads(diccionario_riqueza=riqueza_oos_por_ratio, fechas_oos=T, perfiles=perfiles, ratios=[r for r in ratios if r != 'SR'])
plot_single_selection_spread(diccionario_riqueza=riqueza_oos_por_ratio, ratios=[r for r in ratios if r != 'SR'], base_strategy=perfiles[0], benchmark='SR')

# II) QUÉ CARTERA DISTRIBUYE MEJOR EL CAPITAL?
allocation_spreads_vs_ew_benchmark = plot_allocation_spreads(diccionario_riqueza=riqueza_oos_por_ratio, fechas_oos=T, ratios=ratios, perfiles=perfiles)
plot_single_allocation_spread(diccionario_riqueza=riqueza_oos_por_ratio, perfiles=perfiles, ratio=ratios[0], benchmark='EW')

# III) EVOLUCIÓN DEL ÍNDICE DE HERFINDAHL (MHI)
mhi_spreads_by_ratio = plot_mhi(mhi_history=mhi_history, fechas_oos=T, ratios=ratios, perfiles=perfiles)

# RESUMEN FINAL: PERFORMANCE DE TODAS LAS ESTRATEGIAS
resumen = []
for ratio, wealth_df in riqueza_oos_por_ratio.items():
    final_row = wealth_df.iloc[-1]
    for allocation_strategy in wealth_df.columns:
        
        riqueza_serie = wealth_df[allocation_strategy]
        maxDD = maxDrawDown(riqueza_serie)
        
        clave_resultados = f"{ratio}_{allocation_strategy}"
        sharpe_val = resultados_oos.loc[clave_resultados, 'Sharpe Ratio Anualizado']
        
        resumen.append({
            'Ratio_Seleccion': ratio,
            'Perfil_Pesos': allocation_strategy,
            'Riqueza_Final': final_row[allocation_strategy],
            'Retorno_Acumulado_%': (final_row[allocation_strategy] - 1.0) * 100,
            'MaxDrawDown_%': maxDD * 100,
            'Sharpe_Ratio': sharpe_val
        })
        
summary_df = pd.DataFrame(resumen)

print("\nRebalanceo medio diario (Costes de transacción):")
print()
for ratio in ratios:
    print(f"   • {ratio:<10}: {rotation_mean[ratio]:.4f} ETFs/día")

print("\n [ 1. RENTABILIDAD ABSOLUTA (RIQUEZA ACUMULADA) ]")

# ORDENAMOS POR RIQUEZA ACUMULADA
summary_ranked_ret = summary_df.sort_values(by='Riqueza_Final', ascending=False).reset_index(drop=True)
top5_ret = summary_ranked_ret.head(5)
bottom5_ret = summary_ranked_ret.tail(5).sort_values(by='Riqueza_Final', ascending=True).reset_index(drop=True)

best_ret = top5_ret.iloc[0]
worst_ret = bottom5_ret.iloc[0]

print("\n   • Top 5 estrategias (Mayor Rendimiento):")
for i in range(5):
    print(f"     {i+1}. {top5_ret.iloc[i]['Ratio_Seleccion']:<12} + {top5_ret.iloc[i]['Perfil_Pesos']:<5} | Rendimiento: {top5_ret.iloc[i]['Retorno_Acumulado_%']:>6.2f}% (DD: {top5_ret.iloc[i]['MaxDrawDown_%']:.2f}%)")

print("\n   • Bottom 5 estrategias (Peor Rendimiento):")
for i in range(5):
    print(f"     {i+1}. {bottom5_ret.iloc[i]['Ratio_Seleccion']:<12} + {bottom5_ret.iloc[i]['Perfil_Pesos']:<5} | Rendimiento: {bottom5_ret.iloc[i]['Retorno_Acumulado_%']:>6.2f}% (DD: {bottom5_ret.iloc[i]['MaxDrawDown_%']:.2f}%)")


print(f"   ► Diferencia de capital (Mejor - Peor): {(best_ret['Riqueza_Final'] / worst_ret['Riqueza_Final'] - 1.0) * 100:.2f}%\n")

print("\n [ 2. PROTECCIÓN DE CAPITAL (MAXIMUM DRAWDOWN) ]")

# ORDENAMOS POR MAXIMO DRAWDOWN (MÁXIMA PÉRDIDA, EL MÁS BAJO ARRIBA)
summary_ranked_dd = summary_df.sort_values(by='MaxDrawDown_%', ascending=False).reset_index(drop=True)
top5_dd = summary_ranked_dd.head(5)
bottom5_dd = summary_ranked_dd.tail(5).sort_values(by='MaxDrawDown_%', ascending=True).reset_index(drop=True)

best_dd = top5_dd.iloc[0]
worst_dd = bottom5_dd.iloc[0]

print("\n   • Top 5 estrategias (Menor Caída Máxima):")
for i in range(5):
    print(f"     {i+1}. {top5_dd.iloc[i]['Ratio_Seleccion']:<12} + {top5_dd.iloc[i]['Perfil_Pesos']:<5} | DD: {top5_dd.iloc[i]['MaxDrawDown_%']:>6.2f}% (Rendimiento: {top5_dd.iloc[i]['Retorno_Acumulado_%']:.2f}%)")

print("\n   • Bottom 5 estrategias (Mayor Destrucción de Capital):")
for i in range(5):
    print(f"     {i+1}. {bottom5_dd.iloc[i]['Ratio_Seleccion']:<12} + {bottom5_dd.iloc[i]['Perfil_Pesos']:<5} | DD: {bottom5_dd.iloc[i]['MaxDrawDown_%']:>6.2f}% (Rendimiento: {bottom5_dd.iloc[i]['Retorno_Acumulado_%']:.2f}%)")

print("\n [ 3. RENTABILIDAD AJUSTADA AL RIESGO (SHARPE RATIO) ]")

# ORDENAMOS POR RATIO DE SHARPE
summary_ranked_sr = summary_df.sort_values(by='Sharpe_Ratio', ascending=False).reset_index(drop=True)
top5_sr = summary_ranked_sr.head(5)
bottom5_sr = summary_ranked_sr.tail(5).sort_values(by='Sharpe_Ratio', ascending=True).reset_index(drop=True)

best_sr = top5_sr.iloc[0]
worst_sr = bottom5_sr.iloc[0]

print("\n   • Top 5 estrategias (Mejor Equilibrio Rentabilidad/Riesgo):")
for i in range(5):
    print(f"     {i+1}. {top5_sr.iloc[i]['Ratio_Seleccion']:<12} + {top5_sr.iloc[i]['Perfil_Pesos']:<5} | Sharpe: {top5_sr.iloc[i]['Sharpe_Ratio']:>6.4f} (Rendimiento: {top5_sr.iloc[i]['Retorno_Acumulado_%']:>6.2f}%, DD: {top5_sr.iloc[i]['MaxDrawDown_%']:.2f}%)")

print("\n   • Bottom 5 estrategias (Peor Equilibrio Rentabilidad/Riesgo):")
for i in range(5):
    print(f"     {i+1}. {bottom5_sr.iloc[i]['Ratio_Seleccion']:<12} + {bottom5_sr.iloc[i]['Perfil_Pesos']:<5} | Sharpe: {bottom5_sr.iloc[i]['Sharpe_Ratio']:>6.4f} (Rendimiento: {bottom5_sr.iloc[i]['Retorno_Acumulado_%']:>6.2f}%, DD: {bottom5_sr.iloc[i]['MaxDrawDown_%']:.2f}%)")

print(f"\n   ► Estrategia más Eficiente : {best_sr['Ratio_Seleccion']} + {best_sr['Perfil_Pesos']}")
print(f"      ↳ Sharpe Ratio : {best_sr['Sharpe_Ratio']:.4f} | Riqueza Final: {best_sr['Riqueza_Final']:.4f} (DD: {best_sr['MaxDrawDown_%']:.2f}%)")

print(f"   ► Estrategia menos Eficiente: {worst_sr['Ratio_Seleccion']} + {worst_sr['Perfil_Pesos']}")
print(f"      ↳ Sharpe Ratio : {worst_sr['Sharpe_Ratio']:.4f} | Riqueza Final: {worst_sr['Riqueza_Final']:.4f} (DD: {worst_sr['MaxDrawDown_%']:.2f}%)")

# ANÁLISIS DE ROBUSTEZ (ERROR DE ESTIMACIÓN)
ratio_robustez = 'SR'
cartera_robustez = 'GMV'
modelo_robustez = W_GMV
ids_matrix_robustez = rankings[ratio_robustez]

# ANÁLISIS DE ROBUSTEZ - TAMAÑO DE VENTANA (M vs M2)
M2 = 2500

print(f"\nIniciando Análisis de Robustez (M={M} vs M={M2})...")

# Definimos un nuevo horizonte (T_rob) que empiece estrictamente en el día M2 (2500)
T_rob = df_rendimientos.index[df_rendimientos.index >= df_rendimientos.index[M2]]
t_indices_rob = df_rendimientos.index.searchsorted(T_rob)

# Recalculamos los rankings EXCLUSIVAMENTE para la ventana de M2500 a partir de esa fecha
rankings_M2500, _ = ROLLING_WINDOW_RANKINGS(df_rendimientos, T_rob[0], K, [ratio_robustez], M=M2)
ranking_rob_df = rankings_M2500[ratio_robustez]

# Alineamos los IDs de M=2500 con las fechas T_rob
t_rob_idx = ranking_rob_df.index.get_indexer(T_rob)
t_anterior_rob = np.maximum(0, t_rob_idx - 1)
ids_matrix_robustez_M2500 = (ranking_rob_df.iloc[t_anterior_rob].values - 1).astype(int)

# Alineamos los IDs que ya teníamos de M=1000 recortando los primeros días que sobran
offset = len(T) - len(T_rob)
ids_matrix_robustez_M1000 = ids_matrix_robustez[offset:]

# Simulamos ambas carteras sobre el MISMO periodo de tiempo (T_rob)
rendimientos_M1000, riqueza_M1000 = allocation_model_with_costs(
    T_rob, t_indices_rob, M, K, rendimientos, ids_matrix_robustez_M1000, modelo_robustez, cartera_robustez, c=c)

rendimientos_M2500, riqueza_M2500 = allocation_model_with_costs(
    T_rob, t_indices_rob, M2, K, rendimientos, ids_matrix_robustez_M2500, modelo_robustez, cartera_robustez, c=c)

# 4. Gráfico Comparativo
plot_robustness_analysis(
    f'Gráfico comparativo (M) para ({cartera_robustez} + {ratio_robustez})',
    {
        f'M = {M}': pd.Series(riqueza_M1000, index=T_rob),
        f'M = {M2}': pd.Series(riqueza_M2500, index=T_rob) 
    }
)

# ANÁLISIS DE ROBUSTEZ - GAMMA / AVERSIÓN AL RIESGO (CARTERA MVS)
rendimientos_MVS_g1, riqueza_MVS_g1 = allocation_model_with_costs(
    T, t_indices, M, K, rendimientos, ids_matrix_robustez, W_MVS, 'MVS', c=c, gamma=1.0)

rendimientos_MVS_g5, riqueza_MVS_g5 = allocation_model_with_costs(
    T, t_indices, M, K, rendimientos, ids_matrix_robustez, W_MVS, 'MVS', c=c, gamma=5.0)

rendimientos_MVS_g10, riqueza_MVS_g10 = allocation_model_with_costs(
    T, t_indices, M, K, rendimientos, ids_matrix_robustez, W_MVS, 'MVS', c=c, gamma=10.0)

# GRÁFICO COMPARACIÓN GAMMA EN LA MVS
plot_robustness_analysis(
    f'Impacto de la Aversión al Riesgo (Gamma) - (MVS + {ratio_robustez})',
    {
        'Gamma = 1': pd.Series(riqueza_MVS_g1, index=T),
        'Gamma = 5': pd.Series(riqueza_MVS_g5, index=T),
        'Gamma = 10': pd.Series(riqueza_MVS_g10, index=T)})

plt.show()

# EXPORTAMOS RESULTADOS A EXCEL
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    
    # RENDIMIENTOS DIARIOS Y RIQUEZA ACUMULADA POR RATIO DE SELECCIÓN
    for ratio in ratios:
        rendimientos_oos_por_ratio[ratio].to_excel(writer, sheet_name=f'Rendimientos_{ratio}')
        riqueza_oos_por_ratio[ratio].to_excel(writer, sheet_name=f'Riqueza_{ratio}')
    
    # SPREADS: COMPARACIÓN DE PERFILES DE PESOS VS BENCHMARK SHARPE
    for perfil, spread_df in selection_spreads_vs_sr_benchmark.items():
        spread_df.to_excel(writer, sheet_name=f'Spread_SR_vs_{perfil}')
    
    # SPREADS: COMPARACIÓN DE RATIOS DE SELECCIÓN VS BENCHMARK EW
    for ratio, spread_df in allocation_spreads_vs_ew_benchmark.items():
        spread_df.to_excel(writer, sheet_name=f'Spread_EW_vs_{ratio}')

    # MHI POR RATIO Y PERFIL
    for ratio, mhi_df in mhi_spreads_by_ratio.items():
        mhi_df.to_excel(writer, sheet_name=f'MHI_{ratio}')
    
    # RESÚMENES FINALES
    summary_df.to_excel(writer, sheet_name='Resumen', index=False)
    resultados_oos.to_excel(writer, sheet_name='Rendimientos_OOS')

print(f"\nCálculos completados exitosamente. Archivo guardado en: {OUTPUT_FILE}")
