"""
PLOTS: GRAFICOS
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martinez
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import scipy.cluster.hierarchy as sch
import seaborn as sns
import math
import numpy as np

# Configuración base para todos los plots
def set_latex_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False
    })

def plot_boxplots(df_metrics: pd.DataFrame):
    set_latex_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    
    metrics_map = {'Volatility': 'Volatilidad', 'Skewness': 'Asimetría', 'Kurtosis': 'Curtosis'}
    
    for ax, (col, title) in zip(axes, metrics_map.items()):
        sns.boxplot(y=df_metrics[col], ax=ax, color="white", width=0.25, linewidth=1.2,
                    boxprops=dict(edgecolor="black", zorder=3),
                    whiskerprops=dict(color="black", linewidth=1.2),
                    capprops=dict(color="black", linewidth=1.2),
                    medianprops=dict(color="black", linewidth=1.8),
                    flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black", alpha=0.5))
        
        ax.set_title(title, fontweight='normal', pad=15)
        ax.set_xticks([])
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)

    plt.tight_layout(w_pad=4.0)
    plt.savefig("boxplots_tfm.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_efficient_frontier(df_efficient_frontier, opt_sharpe, opt_vol, mean_returns, cov_matrix):
    set_latex_style()
    def get_metrics(w): return float(w.T @ mean_returns), float(np.sqrt(w.T @ cov_matrix @ w))

    ret_sharpe, vol_sharpe = get_metrics(opt_sharpe.x)
    ret_vol, vol_vol = get_metrics(opt_vol.x)

    fig, ax = plt.subplots(figsize=(10, 6))
    asset_volatilities = np.sqrt(np.diag(cov_matrix))
    ax.scatter(asset_volatilities, mean_returns, marker='o', color='black', s=15, alpha=0.6, label='ETFs Individuales')
    ax.plot(df_efficient_frontier['Volatility'], df_efficient_frontier['Return'], color='black', linewidth=1.5, label='Frontera Eficiente')
    ax.scatter(vol_sharpe, ret_sharpe, marker='D', color='black', s=80, zorder=5, label='Max Sharpe')
    ax.scatter(vol_vol, ret_vol, marker='s', facecolors='white', edgecolors='black', s=80, zorder=5, label='Min Var')

    ax.set_xlabel('Volatilidad Anualizada')
    ax.set_ylabel('Retorno Esperado Anualizado')
    ax.legend(loc='best', frameon=False)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("efficient_frontier.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_cumulative_returns(diccionario_riqueza: dict):
    set_latex_style()
    metrics_map = {
        'SR': 'SR', 'SR_corr': 'SR*', 'MSR': 'MSR', 'VaRR1': 'VaRR$_{1\%}$', 
        'VaRR5': 'VaRR$_{5\%}$', 'VaRR10': 'VaRR$_{10\%}$', 'Omega': 'Omega', 
        'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': 'GR$_1$', 'GR5': 'GR$_5$'
    }

    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 18), squeeze=False)
    axes = axes.flatten()

    for idx, (raw_title, values) in enumerate(diccionario_riqueza.items()):
        ax = axes[idx]
        spread_bruta = (values[0] - values[1])
        spread_neta = (values[2] - values[1])

        ax.plot(spread_bruta, label='Exceso Bruto', color='black', linestyle='-', linewidth=1.1)
        ax.plot(spread_neta, label='Exceso Neto', color='black', linestyle=':', linewidth=1.6)
        ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
        
        ax.set_title(metrics_map.get(raw_title, raw_title), fontsize=13, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, frameon=False)

    plt.tight_layout(pad=4.0)
    plt.savefig("cumulative_spread_returns.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_turnover_frequency(diccionario_frecuencia: dict, save_path: str = "turnover_frequency.pdf"):
    set_latex_style()
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    axes = axes.flatten()
    
    metrics_map = {
        'SR': 'SR', 'SR_corr': 'SR*', 'MSR': 'MSR', 'VaRR1': 'VaRR$_{1\%}$', 
        'VaRR5': 'VaRR$_{5\%}$', 'VaRR10': 'VaRR$_{10\%}$', 'Omega': 'Omega', 
        'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': 'GR$_1$', 'GR5': 'GR$_5$'
    }

    for i, (col, title) in enumerate(metrics_map.items()):
        ax = axes[i]
        serie = diccionario_frecuencia[col]
        ax.bar(serie.index, serie.values, color="#404040", edgecolor="white", width=1.0)
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-0.5, 20.5)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_robustness_analysis(titulo: str, lineas: dict):
    set_latex_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for nombre, serie in lineas.items():
        ax.plot(serie, label=nombre, linewidth=1.5)

    ax.set_title(titulo)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='best', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("robustness_analysis.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_selection_spreads(diccionario_riqueza, fechas_oos, perfiles, ratios, benchmark='SR'):
    set_latex_style()
    cmap = plt.get_cmap('tab20')
    
    rows, cols = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 18))
    axes = axes.flatten()

    for i, base_strategy in enumerate(perfiles):
        ax = axes[i]
        riqueza_base = diccionario_riqueza[benchmark][base_strategy]
        for j, ratio in enumerate(ratios):
            spread = diccionario_riqueza[ratio][base_strategy] - riqueza_base
            ax.plot(spread.index, spread.values, linewidth=1.2, color=cmap(j), label=f'{ratio}-{benchmark}')

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'Base: {base_strategy}', fontweight='bold')
        ax.legend(fontsize=7, loc='upper left', ncol=2, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("selection_spreads.pdf", format='pdf', bbox_inches='tight', transparent=True)
    return {}

def plot_allocation_spreads(diccionario_riqueza, fechas_oos, ratios, perfiles, benchmark='EW'):
    set_latex_style()
    cmap = plt.get_cmap('tab20')
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    axes = axes.flatten()

    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        riqueza_base = diccionario_riqueza[ratio][benchmark]
        for j, strat in enumerate(perfiles):
            spread = diccionario_riqueza[ratio][strat] - riqueza_base
            ax.plot(spread.index, spread.values, linewidth=1.2, color=cmap(j), label=f'{strat}-{benchmark}')

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'Ratio: {ratio}', fontweight='bold')
        ax.legend(fontsize=7, loc='lower left', ncol=2, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("allocation_spreads.pdf", format='pdf', bbox_inches='tight', transparent=True)
    return {}

def plot_mhi(mhi_history, fechas_oos, ratios, perfiles):
    set_latex_style()
    cmap = plt.get_cmap('tab20')
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    axes = axes.flatten()

    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        for j, perfil in enumerate(perfiles):
            ax.plot(fechas_oos, mhi_history[ratio][perfil], linewidth=1.2, color=cmap(j), label=perfil)

        ax.set_title(f'MHI - {ratio}', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc='lower left', ncol=2, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("mhi_evolution.pdf", format='pdf', bbox_inches='tight', transparent=True)
    return {}

def plot_dendrogram(link: np.ndarray, labels: list, title: str, save_path: str = "dendrogram.pdf"):
    set_latex_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sch.dendrogram(link, labels=labels, leaf_rotation=90, ax=ax, color_threshold=0, above_threshold_color='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_matrix_heatmap(matrix: pd.DataFrame, title: str, save_path: str = "matrix_heatmap.pdf"):
    set_latex_style()
    plt.figure(figsize=(8, 6))
    plt.pcolor(matrix, cmap='jet')
    plt.colorbar()
    plt.yticks(np.arange(0.5, matrix.shape[0] + 0.5), matrix.index)
    plt.xticks(np.arange(0.5, matrix.shape[1] + 0.5), matrix.columns, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_coincidence_heatmap(matrix: pd.DataFrame, save_path: str = "coincidence_heatmap.pdf"):
    set_latex_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap=sns.light_palette("#1f77b4", as_cmap=True),
                vmin=0, vmax=100, cbar_kws={"label": "Coincidencia (%)"}, square=True, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_survival_heatmap(all_stats_dict):
    set_latex_style()
    metrics_order = ['SR', 'SR_corr', 'MSR', 'Sortino', 'Kappa3', 'Omega', 'VaRR1', 'VaRR5', 'VaRR10', 'UPR', 'GR1', 'GR5']
    labels = {'VaRR1': 'VaRR$_{1\%}$', 'VaRR5': 'VaRR$_{5\%}$', 'VaRR10': 'VaRR$_{10\%}$', 'GR1': 'GR$_1$', 'GR5': 'GR$_5$', 'SR_corr': 'SR*'}

    survival_matrix = [all_stats_dict[m].head(5)['Survival_Rate'].values * 100 for m in metrics_order]
    ticker_matrix = [all_stats_dict[m].head(5)['Ticker'].values for m in metrics_order]

    survival_df = pd.DataFrame(survival_matrix, index=[labels.get(m, m) for m in metrics_order], columns=['1º', '2º', '3º', '4º', '5º'])

    plt.figure(figsize=(11, 8))
    sns.heatmap(survival_df, annot=np.array(ticker_matrix), fmt="", cmap="Blues", linewidths=0.5, annot_kws={"size": 10, "fontweight": "bold"})
    
    plt.tight_layout()
    plt.savefig("survival_heatmap.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()