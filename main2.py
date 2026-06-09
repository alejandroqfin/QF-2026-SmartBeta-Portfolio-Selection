"""
main2.py
PARTE II: GESTIÓN DE CARTERAS, RIESGO Y PERFORMANCE
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martínez

Nota:
Este código procede y requiere ejecutar previamente el código de la PARTE I [main1.py] 
Por tanto hereda las variables definidas en el código anterior [screening_artifacts.joblib]
para estudiar el impacto de la gestión del capital.
"""

import numpy as np
import pandas as pd
import joblib
from metrics import risk_contribution, allocation_model_with_costs, MHI, maxDrawDown
from plots import plot_allocation_spreads, plot_selection_spreads, plot_mhi, plot_dendrogram, plot_matrix_heatmap, plot_dendrogram_heatmap, plot_single_allocation_spread, plot_single_selection_spread
from portfolios import W_EW, W_VT, W_RRT, W_GMV, W_ERC, W_MVS, W_HRP, W_HERC
from hrp import HRP, HERC

OUTPUT_FILE = "100_ETFs_Allocation.xlsx"
VARIABLES_FILE = "screening_artifacts.joblib"
VERTICAL_DATE = pd.to_datetime("2024-12-19")

# RANKINGS DE SELECCIÓN (Top K por ratio) Y PERFILES DE PESOS
ratios = ['SR', 'SR_corr', 'MSR', 'VaRR1', 'VaRR5', 'VaRR10', 'Omega', 'UPR', 'Kappa3', 'Sortino', 'GR1', 'GR5']
perfiles = ['EW', 'VT', 'RRT', 'GMV', 'ERC', 'MVS', 'HRP', 'HERC']

# RECUPERAMOS VARIABLES
variables = joblib.load(VARIABLES_FILE)

# COSTES DE TRANSACCIÓN
c = variables['c']

# VARIABLES
K = variables['K']
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

print("DATOS")
print("\n Configuración de la Cartera ")
print(f"  • Universo de ETFs (N):      {len(tickers)}")
print(f"  • Activos en cartera (K):   {K}")
print(f"  • Costes de transacción (c): {c}")

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

# ALGORITMO HERC
w_HERC, links_vertical_HERC, Sigma_quasidiag_HERC = HERC(Sigma, labels=etfs_vertical)

# CORRELACIONES CUASIDIAGONALIZADAS
corr_quasidiag_HRP = corr_matrix.loc[Sigma_quasidiag_HRP.index, Sigma_quasidiag_HRP.columns]
corr_quasidiag_HERC = corr_matrix.loc[Sigma_quasidiag_HERC.index, Sigma_quasidiag_HERC.columns]

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
w_HERC = w_HERC                                     # (K x 1)

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

# MÉTRICAS DE LOS MODELOS IN-SAMPLE
in_sample_portfolios = [
    ("EQUALLYWEIGHTED (EW)", mu_EW, var_EW, SR_EW, w_EW, RC_abs_EW, RC_rel_EW),
    ("VOLATILITY TIMING (VT)", mu_VT, var_VT, SR_VT, w_VT, RC_abs_VT, RC_rel_VT),
    ("REWARD-TO-RISK TIMING (RRT)", mu_RRT, var_RRT, SR_RRT, w_RRT, RC_abs_RRT, RC_rel_RRT),
    ("GLOBAL MINIMUM VARIANCE (GMV)", mu_GMV, var_GMV, SR_GMV, w_GMV, RC_abs_GMV, RC_rel_GMV),
    ("EQUAL RISK CONTRIBUTION (ERC)", mu_ERC, var_ERC, SR_ERC, w_ERC, RC_abs_ERC, RC_rel_ERC),
    ("MEAN-VARIANCE-SKEWNESS (MVS)", mu_MVS, var_MVS, SR_MVS, w_MVS, RC_abs_MVS, RC_rel_MVS),
    ("HIERARCHICAL RISK PARITY (HRP)", mu_HRP, var_HRP, SR_HRP, w_HRP, RC_abs_HRP, RC_rel_HRP),
    ("HIERARCHICAL EQUAL RISK CONTRIBUTION (HERC)", mu_HERC, var_HERC, SR_HERC, w_HERC, RC_abs_HERC, RC_rel_HERC)
]

for idx, (name, mu_val, var_val, sr_val, w_val, rc_abs_val, rc_rel_val) in enumerate(in_sample_portfolios, 1):
    print(f"\n{idx}. {name}")
    print(f"   ► Media                    : {mu_val:.8f}")
    print(f"   ► Varianza                 : {var_val:.10f}")
    print(f"   ► Volatilidad              : {np.sqrt(var_val):.4f}")
    print(f"   ► Ratio de Sharpe          : {sr_val:.4f}")
    print(f"   ► Pesos                    : {np.round(w_val * 100, 2)}%")
    print(f"   ► RC Absoluta              : {np.round(rc_abs_val, 4)}")
    print(f"   ► RC Relativa (%)          : {np.round(rc_rel_val * 100, 2)}%")

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

# IV) RESUMEN FINAL: PERFORMANCE DE TODAS LAS ESTRATEGIAS
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

# REBALANCEO MEDIO DIARIO
print("\nRebalanceo medio diario (Costes de transacción):")
print()
for ratio in ratios:
    print(f"   • {ratio:<10}: {rotation_mean[ratio]:.4f} ETFs/día")

# PANEL A: RIQUEZA ACUMULADA
print("\n [ PANEL A: RIQUEZA ACUMULADA (RIQUEZA ACUMULADA) ]")

# ORDENAMOS DE MAYOR A MENOR
summary_ranked_ret = summary_df.sort_values(by='Riqueza_Final', ascending=False).reset_index(drop=True)
top5_ret = summary_ranked_ret.head(5)
bottom5_ret = summary_ranked_ret.tail(5).sort_values(by='Riqueza_Final', ascending=True).reset_index(drop=True)

print("\n    • Top 5 estrategias (Mayor Rendimiento):")
for idx in range(5):
    row = top5_ret.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | Rendimiento: {row['Retorno_Acumulado_%']:>6.2f}% (DD: -{row['MaxDrawDown_%']:.2f}%) [Sharpe: {row['Sharpe_Ratio']:.4f}]")

print("\n    • Bottom 5 estrategias (Peor Rendimiento):")
for idx in range(5):
    row = bottom5_ret.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | Rendimiento: {row['Retorno_Acumulado_%']:>6.2f}% (DD: -{row['MaxDrawDown_%']:.2f}%) [Sharpe: {row['Sharpe_Ratio']:.4f}]")

# PANEL B: MAXIMO DRAWDOWN (MÁXIMA PÉRDIDA)
print("\n [ PANEL B: PROTECCIÓN DE CAPITAL (MAXIMUM DRAWDOWN) ]")

summary_ranked_dd_best = summary_df.sort_values(by='MaxDrawDown_%', ascending=False).reset_index(drop=True)
top5_dd = summary_ranked_dd_best.head(5)
summary_ranked_dd_worst = summary_df.sort_values(by='MaxDrawDown_%', ascending=True).reset_index(drop=True)
bottom5_dd = summary_ranked_dd_worst.head(5)

print("\n    • Top 5 estrategias (Menor Caída Máxima - DE MEJOR A PEOR):")
for idx in range(5):
    row = top5_dd.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | DD: {row['MaxDrawDown_%']:>6.2f}% (Rendimiento: {row['Retorno_Acumulado_%']:.2f}%) [Sharpe: {row['Sharpe_Ratio']:.4f}]")

print("\n    • Bottom 5 estrategias (Mayor Destrucción de Capital - DE PEOR A MEJOR):")
for idx in range(5):
    row = bottom5_dd.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | DD: {row['MaxDrawDown_%']:>6.2f}% (Rendimiento: {row['Retorno_Acumulado_%']:.2f}%) [Sharpe: {row['Sharpe_Ratio']:.4f}]")

# PANEL C: RENTABILIDAD AJUSTADA AL RIESGO (SHARPE RATIO)
print("\n [ PANEL C: RENTABILIDAD AJUSTADA AL RIESGO (SHARPE RATIO) ]")

summary_ranked_sr = summary_df.sort_values(by='Sharpe_Ratio', ascending=False).reset_index(drop=True)
top5_sr = summary_ranked_sr.head(5)
bottom5_sr = summary_ranked_sr.tail(5).sort_values(by='Sharpe_Ratio', ascending=True).reset_index(drop=True)
best_sr = top5_sr.iloc[0]
worst_sr = bottom5_sr.iloc[0]

print("\n    • Top 5 estrategias (Mejor Equilibrio Rentabilidad/Riesgo):")
for idx in range(5):
    row = top5_sr.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | Sharpe: {row['Sharpe_Ratio']:>6.4f} (Rendimiento: {row['Retorno_Acumulado_%']:>6.2f}%, DD: -{row['MaxDrawDown_%']:.2f}%)")

print("\n    • Bottom 5 estrategias (Peor Equilibrio Rentabilidad/Riesgo):")
for idx in range(5):
    row = bottom5_sr.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | Sharpe: {row['Sharpe_Ratio']:>6.4f} (Rendimiento: {row['Retorno_Acumulado_%']:>6.2f}%, DD: -{row['MaxDrawDown_%']:.2f}%)")

print(f"\n    ► Estrategia más Eficiente : {best_sr['Ratio_Seleccion']} + {best_sr['Perfil_Pesos']}")
print(f"       ↳ Sharpe Ratio : {best_sr['Sharpe_Ratio']:.4f} | Riqueza Final: {best_sr['Riqueza_Final']:.4f} (DD: -{best_sr['MaxDrawDown_%']:.2f}%)")

print(f"    ► Estrategia menos Eficiente: {worst_sr['Ratio_Seleccion']} + {worst_sr['Perfil_Pesos']}")
print(f"       ↳ Sharpe Ratio : {worst_sr['Sharpe_Ratio']:.4f} | Riqueza Final: {worst_sr['Riqueza_Final']:.4f} (DD: -{worst_sr['MaxDrawDown_%']:.2f}%)")

# ANEXO C.2. - ANÁLISIS DE SENSIBILIDAD A LOS COSTES DE TRANSACCIÓN (c = 0.0000)
print(" ANEXO C.2. - ANÁLISIS DE SENSIBILIDAD (c = 0.0000)")
rendimientos_oos_c0 = {}
riqueza_oos_c0 = {}
for ratio in ratios:
    print(f"\n Ratio de Selección: {ratio}")
    rendimientos_oos_c0[ratio] = {}
    riqueza_oos_c0[ratio] = {}
    ids_matrix = rankings[ratio]
    
    for perfil, model in perfiles_de_pesos:
        print(f"  -> Optimizando pesos según: {perfil}")
        r_neto, riq_neta = allocation_model_with_costs(
            T, t_indices, M, K, rendimientos, ids_matrix, model, perfil, c=0.0, save_mhi=None)
        rendimientos_oos_c0[ratio][perfil] = r_neto
        riqueza_oos_c0[ratio][perfil] = riq_neta

rendimientos_oos_por_ratio_c0 = {}
riqueza_oos_por_ratio_c0 = {}

for ratio in ratios:
    rendimientos_oos_por_ratio_c0[ratio] = pd.DataFrame(
        {perfil: rendimientos_oos_c0[ratio][perfil] for perfil in perfiles}, index=T
    )
    riqueza_oos_por_ratio_c0[ratio] = pd.DataFrame(
        {perfil: riqueza_oos_c0[ratio][perfil] for perfil in perfiles}, index=T
    )

df_rendimientos_estrategias_c0 = pd.concat(rendimientos_oos_por_ratio_c0, axis=1)
df_rendimientos_estrategias_c0.columns = [f"{ratio}_{perfil}" for ratio, perfil in df_rendimientos_estrategias_c0.columns]

# CALCULO DE MÉTRICAS ANUALIZADAS SIN COSTES
resultados_oos_c0 = pd.DataFrame({
    'Rentabilidad Anualizada': df_rendimientos_estrategias_c0.mean() * annual_factor,
    'Volatilidad Anualizada': df_rendimientos_estrategias_c0.std() * np.sqrt(annual_factor),
    'Sharpe Ratio Anualizado': (df_rendimientos_estrategias_c0.mean() / df_rendimientos_estrategias_c0.std()) * np.sqrt(annual_factor),
    'Asimetría (Skewness)': df_rendimientos_estrategias_c0.skew(),
}).sort_values(by='Sharpe Ratio Anualizado', ascending=False)

# RESUMEN FINAL PARA LOS PANELES DE ROBUSTEZ
resumen_c0 = []
for ratio, wealth_df in riqueza_oos_por_ratio_c0.items():
    final_row = wealth_df.iloc[-1]
    for allocation_strategy in wealth_df.columns:
        riqueza_serie = wealth_df[allocation_strategy]
        maxDD = maxDrawDown(riqueza_serie)
        
        clave_resultados = f"{ratio}_{allocation_strategy}"
        sharpe_val = resultados_oos_c0.loc[clave_resultados, 'Sharpe Ratio Anualizado']
        
        resumen_c0.append({
            'Ratio_Seleccion': ratio,
            'Perfil_Pesos': allocation_strategy,
            'Riqueza_Final': final_row[allocation_strategy],
            'Retorno_Acumulado_%': (final_row[allocation_strategy] - 1.0) * 100,
            'MaxDrawDown_%': maxDD * 100,
            'Sharpe_Ratio': sharpe_val
        })
        
summary_df_c0 = pd.DataFrame(resumen_c0)

# TABLA DE RESULTADOS FINALES PARA c = 0.0000
print(" [ ROBUSTEZ: PANEL A: RIQUEZA ACUMULADA (c = 0.0) ]")

summary_ranked_ret_c0 = summary_df_c0.sort_values(by='Riqueza_Final', ascending=False).reset_index(drop=True)
top5_ret_c0 = summary_ranked_ret_c0.head(5)
bottom5_ret_c0 = summary_ranked_ret_c0.tail(5).sort_values(by='Riqueza_Final', ascending=True).reset_index(drop=True)
best_ret_c0 = top5_ret_c0.iloc[0]
worst_ret_c0 = bottom5_ret_c0.iloc[0]

print("\n    • Top 5 estrategias (Mayor Rendimiento - Sin Costes):")
for idx in range(5):
    row = top5_ret_c0.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | Rendimiento: {row['Retorno_Acumulado_%']:>6.2f}% (DD: -{row['MaxDrawDown_%']:.2f}%) [Sharpe: {row['Sharpe_Ratio']:.4f}]")

print("\n    • Bottom 5 estrategias (Peor Rendimiento - Sin Costes):")
for idx in range(5):
    row = bottom5_ret_c0.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | Rendimiento: {row['Retorno_Acumulado_%']:>6.2f}% (DD: -{row['MaxDrawDown_%']:.2f}%) [Sharpe: {row['Sharpe_Ratio']:.4f}]")


print(" [ ROBUSTEZ: PANEL B: PROTECCIÓN DE CAPITAL (MAXIMUM DRAWDOWN c = 0.0) ]")

summary_ranked_dd_best_c0 = summary_df_c0.sort_values(by='MaxDrawDown_%', ascending=True).reset_index(drop=True)
top5_dd_c0 = summary_ranked_dd_best_c0.head(5)
summary_ranked_dd_worst_c0 = summary_df_c0.sort_values(by='MaxDrawDown_%', ascending=False).reset_index(drop=True)
bottom5_dd_c0 = summary_ranked_dd_worst_c0.head(5)

print("\n    • Top 5 estrategias (Menor Caída Máxima - Sin Costes):")
for idx in range(5):
    row = top5_dd_c0.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | DD: -{row['MaxDrawDown_%']:>6.2f}% (Rendimiento: {row['Retorno_Acumulado_%']:.2f}%) [Sharpe: {row['Sharpe_Ratio']:.4f}]")

print("\n    • Bottom 5 estrategias (Mayor Destrucción de Capital - Sin Costes):")
for idx in range(5):
    row = bottom5_dd_c0.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | DD: -{row['MaxDrawDown_%']:>6.2f}% (Rendimiento: {row['Retorno_Acumulado_%']:.2f}%) [Sharpe: {row['Sharpe_Ratio']:.4f}]")


print(" [ ROBUSTEZ: PANEL C: RENTABILIDAD AJUSTADA AL RIESGO (SHARPE RATIO c = 0.0) ]")

summary_ranked_sr_c0 = summary_df_c0.sort_values(by='Sharpe_Ratio', ascending=False).reset_index(drop=True)
top5_sr_c0 = summary_ranked_sr_c0.head(5)
bottom5_sr_c0 = summary_ranked_sr_c0.tail(5).sort_values(by='Sharpe_Ratio', ascending=True).reset_index(drop=True)
best_sr_c0 = top5_sr_c0.iloc[0]
worst_sr_c0 = bottom5_sr_c0.iloc[0]

print("\n    • Top 5 estrategias (Mejor Equilibrio Rentabilidad/Riesgo - Sin Costes):")
for idx in range(5):
    row = top5_sr_c0.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | Sharpe: {row['Sharpe_Ratio']:>6.4f} (Rendimiento: {row['Retorno_Acumulado_%']:>6.2f}%, DD: -{row['MaxDrawDown_%']:.2f}%)")

print("\n    • Bottom 5 estrategias (Peor Equilibrio Rentabilidad/Riesgo - Sin Costes):")
for idx in range(5):
    row = bottom5_sr_c0.iloc[idx]
    print(f"      {idx+1}. {row['Ratio_Seleccion']:<12} + {row['Perfil_Pesos']:<5} | Sharpe: {row['Sharpe_Ratio']:>6.4f} (Rendimiento: {row['Retorno_Acumulado_%']:>6.2f}%, DD: -{row['MaxDrawDown_%']:.2f}%)")

print(f"\n    ► [c=0] Estrategia más Eficiente : {best_sr_c0['Ratio_Seleccion']} + {best_sr_c0['Perfil_Pesos']}")
print(f"        ↳ Sharpe Ratio : {best_sr_c0['Sharpe_Ratio']:.4f} | Riqueza Final: {best_sr_c0['Riqueza_Final']:.4f} (DD: -{best_sr_c0['MaxDrawDown_%']:.2f}%)")

print(f"    ► [c=0] Estrategia menos Eficiente: {worst_sr_c0['Ratio_Seleccion']} + {worst_sr_c0['Perfil_Pesos']}")
print(f"        ↳ Sharpe Ratio : {worst_sr_c0['Sharpe_Ratio']:.4f} | Riqueza Final: {worst_sr_c0['Riqueza_Final']:.4f} (DD: -{worst_sr_c0['MaxDrawDown_%']:.2f}%)")

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
