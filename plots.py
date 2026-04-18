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


def plot_boxplots(df_metrics: pd.DataFrame):
    plt.rcParams['font.family'] = 'serif'
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    metrics = ['Volatility', 'Skewness', 'Kurtosis']
    
    for ax, metric in zip(axes, metrics):
        sns.boxplot(
            y=df_metrics[metric],
            ax=ax,
            color="white", 
            width=0.2, 
            linewidth=1.2,
            boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black", markersize=4) 
        )
        
        ax.set_title(metric, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks([])
        ax.set_ylabel('')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(False)

    plt.tight_layout(w_pad=3.0) 
    plt.savefig("boxplots_tfm.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_efficient_frontier(df_efficient_frontier, opt_sharpe, opt_vol, mean_returns, cov_matrix, risk_free_rate=0.0):
    
    def get_metrics(w):
        return float(w.T @ mean_returns), float(np.sqrt(w.T @ cov_matrix @ w))

    ret_sharpe, vol_sharpe = get_metrics(opt_sharpe.x)
    ret_vol, vol_vol = get_metrics(opt_vol.x)

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(10, 6))

    asset_volatilities = np.sqrt(np.diag(cov_matrix))
    ax.scatter(asset_volatilities, mean_returns, marker='o', color='black', s=15, alpha=0.6, label='ETFs Individuales')

    ax.plot(df_efficient_frontier['Volatility'], df_efficient_frontier['Return'], color='black', linestyle='-', linewidth=1.5, label='Frontera Eficiente')

    ax.scatter(vol_sharpe, ret_sharpe, marker='D', color='black', s=80, zorder=5, label='Cartera Tangente (Max Sharpe)')
    ax.scatter(vol_vol, ret_vol, marker='s', facecolors='white', edgecolors='black', s=80, linewidth=1.2, zorder=5, label='Cartera Mínima Varianza')

    ax.set_xlabel('Volatilidad Anualizada', fontsize=12)
    ax.set_ylabel('Retorno Esperado Anualizado', fontsize=12)
    ax.set_title('Frontera Eficiente de Markowitz', fontsize=14, fontweight='bold', pad=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    ax.legend(loc='best', fontsize=10, frameon=False)
    
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()
    
def plot_cumulative_returns(diccionario_riqueza: dict):
    
    items = list(diccionario_riqueza.items())

    if len(items) <= 12:
        cols, rows = 3, 4
    else:
        cols = 3
        rows = math.ceil(len(items) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, (subplot_title, values) in enumerate(items):

        serie_bruta = values[0]  
        serie_benchmark = values[1]  
        serie_neta = values[2]  
        
        ax = axes[idx]
        
        spread_bruta = (serie_bruta - serie_benchmark).copy()
        spread_neta = (serie_neta - serie_benchmark).copy()

        if len(spread_bruta) > 0:
            spread_bruta = spread_bruta - spread_bruta.iloc[0]
        if len(spread_neta) > 0:
            spread_neta = spread_neta - spread_neta.iloc[0]

        ax.plot(spread_bruta, label='Rolling (sin costes) - Buy & Hold', 
                color='black', linestyle='-', linewidth=1.2, zorder=3)
        
        ax.plot(spread_neta, label='Rolling (con costes) - Buy & Hold', 
                color='black', linestyle=':', linewidth=1.8, zorder=4)
        
        ax.axhline(0, color='#cccccc', linestyle='-', linewidth=1.0, zorder=1)
        
        ax.set_title(f'{subplot_title}', fontsize=12, fontweight='bold') 
        
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#888888')
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', alpha=0.0) 
        
        ax.legend(fontsize=9, loc='lower left', frameon=False)

    for idx in range(len(items), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show(block=True)


def plot_turnover_frequency(diccionario_frecuencia: dict, save_path: str = None):
    
    items = list(diccionario_frecuencia.items())
    
    if len(items) <= 12:
        cols, rows = 3, 4
    else:
        cols = 3
        rows = math.ceil(len(items) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, (subplot_title, freq_series) in enumerate(items):
        ax = axes[idx]
        
        ax.bar(freq_series.index, freq_series.values, 
               color='#404040', alpha=0.85, edgecolor='black', linewidth=0.8)
        
        ax.set_title(f'{subplot_title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Nuevos ETFs por día', fontsize=10)
        ax.set_xticks(freq_series.index)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#888888')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', alpha=0.0)

    for idx in range(len(items), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    plt.show(block=True)


def plot_robustness_analysis(titulo: str, lineas: dict):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    for nombre, serie in lineas.items():
        ax.plot(serie, label=nombre, linewidth=1.5)

    ax.set_title(titulo)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='best', frameon=False)

    plt.tight_layout()
    plt.show(block=False)

def plot_selection_spreads(
    diccionario_riqueza: dict, 
    fechas_oos: pd.DatetimeIndex,
    perfiles: list,
    ratios: list,
    benchmark: str = 'SR'
) -> dict:
    
    n_strats = len(perfiles)
    fig_spreads_by_base = {}

    cmap_name = 'tab10' if len(ratios) <= 10 else 'tab20'
    cmap = plt.get_cmap(cmap_name)
    ratio_colors = [cmap(i) for i in range(len(ratios))]

    cols = 2
    rows = max(1, math.ceil(n_strats / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = axes.flatten()

    for i, base_strategy in enumerate(perfiles):
        ax = axes[i]
        spreads_base = pd.DataFrame(index=fechas_oos)
        
        riqueza_base = diccionario_riqueza[benchmark][base_strategy]

        for j, ratio in enumerate(ratios):
            spread = diccionario_riqueza[ratio][base_strategy] - riqueza_base
            col_name = f'{ratio}_VS_{benchmark}'
            spreads_base[col_name] = spread
            
            ax.plot(
                spread.index,
                spread.values,
                linewidth=1.5,
                color=ratio_colors[j],
                label=f'{ratio} - {benchmark}'
            )

        fig_spreads_by_base[base_strategy] = spreads_base
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)
        ax.set_title(f'Base de pesos: {base_strategy}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spread (EUR)', fontsize=10)
        
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#888888')
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', alpha=0.0)
        ax.legend(fontsize=9, loc='upper left', frameon=False)

    for i in range(n_strats, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    return fig_spreads_by_base


def plot_allocation_spreads(
    diccionario_riqueza: dict, 
    fechas_oos: pd.DatetimeIndex,
    ratios: list,
    perfiles: list,
    benchmark: str = 'EW'
) -> dict:

    n_ratios = len(ratios)
    fig_spreads_all = {}

    cmap_name = 'tab10' if len(perfiles) <= 10 else 'tab20'
    cmap = plt.get_cmap(cmap_name)
    strat_colors = [cmap(i) for i in range(len(perfiles))]

    if n_ratios == 12:
        cols, rows = 4, 3
    else:
        cols = 3
        rows = max(1, math.ceil(n_ratios / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        riqueza_ratio = diccionario_riqueza[ratio]
        riqueza_base = riqueza_ratio[benchmark]

        spreads_ratio = pd.DataFrame(index=fechas_oos)
        
        for j, strat in enumerate(perfiles):
            spread = riqueza_ratio[strat] - riqueza_base
            col_name = f'{strat}_VS_{benchmark}'
            spreads_ratio[col_name] = spread
            
            ax.plot(
                spread.index,
                spread.values,
                linewidth=1.5,
                color=strat_colors[j],
                label=f'{strat} - {benchmark}'
            )

        fig_spreads_all[ratio] = spreads_ratio
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)
        ax.set_title(f'Ratio: {ratio}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spread (EUR)', fontsize=10)
        
        # Diseño minimalista y limpieza de fechas
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#888888')
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', alpha=0.0)
        ax.legend(fontsize=9, loc='lower left', frameon=False)

    for i in range(n_ratios, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    return fig_spreads_all


def plot_mhi(
    mhi_history: dict,
    fechas_oos: pd.DatetimeIndex,
    ratios: list,
    perfiles: list,
) -> dict:

    n_ratios = len(ratios)
    mhi_by_ratio_df = {}

    cmap_name = 'tab10' if len(perfiles) <= 10 else 'tab20'
    cmap = plt.get_cmap(cmap_name)
    perfil_colors = [cmap(i) for i in range(len(perfiles))]

    if n_ratios == 12:
        cols, rows = 4, 3
    else:
        cols = 3
        rows = max(1, math.ceil(n_ratios / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows), squeeze=False)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        ratio_df = pd.DataFrame(index=fechas_oos)

        for j, perfil in enumerate(perfiles):
            serie = mhi_history[ratio][perfil]
            ratio_df[perfil] = serie
            ax.plot(
                fechas_oos,
                serie,
                linewidth=1.5,
                color=perfil_colors[j],
                label=perfil
            )

        mhi_by_ratio_df[ratio] = ratio_df

        ax.set_title(f'MHI - {ratio}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MHI', fontsize=10)
        ax.set_ylim(0.0, 1.05)
        
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#888888')
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', alpha=0.0)
        ax.legend(fontsize=9, loc='lower left', frameon=False)

    for idx in range(n_ratios, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    return mhi_by_ratio_df

def plot_dendrogram(link: np.ndarray, labels: list, title: str):
    """
    Dibuja el dendrograma jerárquico.
    """
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(10, 5))

    sch.dendrogram(
        link, 
        labels=labels, 
        leaf_rotation=90, 
        ax=ax, 
        color_threshold=0, 
        above_threshold_color='black'
    )

    # El título ahora es dinámico y depende de lo que le inyectes al llamar a la función
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

def plot_matrix_heatmap(matrix: pd.DataFrame, title: str):
    """Genera un mapa de calor para una matriz cuadrada (correlación o covarianza)."""
    plt.figure(figsize=(8, 6))
    plt.pcolor(matrix, cmap='jet')
    plt.colorbar()
    plt.yticks(np.arange(0.5, matrix.shape[0] + 0.5), matrix.index)
    plt.xticks(np.arange(0.5, matrix.shape[1] + 0.5), matrix.columns, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr: pd.DataFrame, title: str):
    plot_matrix_heatmap(corr, title)