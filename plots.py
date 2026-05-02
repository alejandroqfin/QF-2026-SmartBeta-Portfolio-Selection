"""
PLOTS: GRAFICOS
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martinez
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import scipy.cluster.hierarchy as sch
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

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
        'SR': 'SR', 'SR_corr': 'SR$_{\\rho}$', 'MSR': 'MSR', 'VaRR1': r'VaRR$_{1\%}$', 
        'VaRR5': r'VaRR$_{5\%}$', 'VaRR10': r'VaRR$_{10\%}$', 'Omega': 'Omega', 
        'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': r'GR$_1$', 'GR5': r'GR$_5$'
    }
    
    cmap = plt.get_cmap('tab20')
    color_dict = {key: cmap(i) for i, key in enumerate(metrics_map.keys())}

    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 18), squeeze=False)
    axes = axes.flatten()

    for idx, (raw_title, values) in enumerate(diccionario_riqueza.items()):
        ax = axes[idx]
        spread_bruta = (values[0] - values[1])
        spread_neta = (values[2] - values[1])
        
        metric_color = color_dict.get(raw_title, 'blue')

        ax.plot(spread_bruta, label='Exceso de Riqueza', color='black', linestyle='--', linewidth=1.2)
        ax.plot(spread_neta, label='Exceso de Riqueza con costes', color=metric_color, linestyle='-', linewidth=1.8)
        
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.8)
        
        ax.set_title(metrics_map.get(raw_title, raw_title), fontsize=13, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, loc='lower left', frameon=False)

    plt.tight_layout(pad=4.0)
    plt.savefig("cumulative_spread_returns.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()
    return {}

def plot_turnover_frequency(diccionario_frecuencia: dict, save_path: str = "turnover_frequency.pdf"):
    set_latex_style()
    
    metrics_map = {
        'SR': 'SR', 'SR_corr': 'SR$_{\\rho}$', 'MSR': 'MSR', 'VaRR1': r'VaRR$_{1\%}$', 
        'VaRR5': r'VaRR$_{5\%}$', 'VaRR10': r'VaRR$_{10\%}$', 'Omega': 'Omega', 
        'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': r'GR$_1$', 'GR5': r'GR$_5$'
    }
    
    cmap = plt.get_cmap('tab20')
    color_dict = {key: cmap(i) for i, key in enumerate(metrics_map.keys())}

    fig, axes = plt.subplots(4, 3, figsize=(12, 14), sharey=True)
    axes = axes.flatten()

    for i, (col, title) in enumerate(metrics_map.items()):
        ax = axes[i]
        serie = diccionario_frecuencia[col]
        
        # Aplicamos el color asignado a la métrica
        ax.bar(serie.index, serie.values, color=color_dict[col], edgecolor="black", linewidth=0.5, width=1.0)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-0.5, 20.5)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()
    return {}

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
    
    metrics_map = {
        'SR': 'SR', 'SR_corr': r'SR$_{\rho}$', 'MSR': 'MSR', 'VaRR1': r'VaRR$_{1\%}$', 
        'VaRR5': r'VaRR$_{5\%}$', 'VaRR10': r'VaRR$_{10\%}$', 'Omega': 'Omega', 
        'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': r'GR$_1$', 'GR5': r'GR$_5$'
    }
    
    cmap_ratios = plt.get_cmap('tab20')
    color_dict = {key: cmap_ratios(i) for i, key in enumerate(metrics_map.keys())}
    
    rows, cols = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 18))
    axes = axes.flatten()

    for i, base_strategy in enumerate(perfiles):
        ax = axes[i]
        riqueza_base = diccionario_riqueza[benchmark][base_strategy]
        
        for j, ratio in enumerate(ratios):
            spread = diccionario_riqueza[ratio][base_strategy] - riqueza_base
            
            label_limpio = metrics_map.get(ratio, ratio)
            color_ratio = color_dict.get(ratio, cmap_ratios(j)) 
            ax.plot(spread.index, spread.values, linewidth=1.2, color=color_ratio, label=label_limpio)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        
        ax.set_title(f'Weight Strategy: {base_strategy} (Spread w.r.t. {benchmark})', fontsize=12, fontweight='bold')
        
        ax.legend(fontsize=8, loc='upper left', ncol=2, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig("selection_spreads.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()
    return {}

def plot_allocation_spreads(diccionario_riqueza, fechas_oos, ratios, perfiles, benchmark='EW'):
    set_latex_style()
    
    metrics_map = {
        'SR': 'SR', 'SR_corr': r'SR$_{\rho}$', 'MSR': 'MSR', 'VaRR1': r'VaRR$_{1\%}$', 
        'VaRR5': r'VaRR$_{5\%}$', 'VaRR10': r'VaRR$_{10\%}$', 'Omega': 'Omega', 
        'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': r'GR$_1$', 'GR5': r'GR$_5$'
    }
    
    cmap_perfiles = plt.get_cmap('tab10') 
    
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    axes = axes.flatten()

    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        riqueza_base = diccionario_riqueza[ratio][benchmark]
        
        for j, strat in enumerate(perfiles):
            spread = diccionario_riqueza[ratio][strat] - riqueza_base
            ax.plot(spread.index, spread.values, linewidth=1.2, color=cmap_perfiles(j), label=strat)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        
        titulo_ratio = metrics_map.get(ratio, ratio)
        
        ax.set_title(f'Performance Strategy: {titulo_ratio} (Spread w.r.t. {benchmark})', fontsize=12, fontweight='bold')
        
        ax.legend(fontsize=8, loc='lower left', ncol=2, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig("allocation_spreads.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()
    return {}

def plot_single_selection_spread(diccionario_riqueza, ratios, base_strategy='EW', benchmark='SR'):
    """
    Extrae un único cuadrante (estrategia base) y lo plotea en formato grande.
    """
    set_latex_style()
    cmap = plt.get_cmap('tab20')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    riqueza_base = diccionario_riqueza[benchmark][base_strategy]
    for j, ratio in enumerate(ratios):
        spread = diccionario_riqueza[ratio][base_strategy] - riqueza_base
        ax.plot(spread.index, spread.values, linewidth=1.5, color=cmap(j), label=f'{ratio}-{benchmark}')

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.legend(fontsize=9, loc='upper left', ncol=3, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"selection_spread_single_{base_strategy}.pdf", format='pdf', bbox_inches='tight', transparent=True)
    return {}

def plot_single_allocation_spread(diccionario_riqueza, perfiles, ratio='SR', benchmark='EW'):
    
    set_latex_style()
    cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(10, 6))
    riqueza_base = diccionario_riqueza[ratio][benchmark]
    for j, strat in enumerate(perfiles):
        spread = diccionario_riqueza[ratio][strat] - riqueza_base
        ax.plot(spread.index, spread.values, linewidth=1.5, color=cmap(j), label=f'{strat}-{benchmark}')

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.legend(fontsize=9, loc='lower left', ncol=4, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"allocation_spread_single_{ratio}.pdf", format='pdf', bbox_inches='tight', transparent=True)
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
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sch.set_link_color_palette(['#1f4e79', '#a50026', '#404040', '#006d2c', '#542788'])

    max_d = link[-1, 2]
    corte_visual = 0.7 * max_d

    sch.dendrogram(
        link, 
        labels=labels, 
        leaf_rotation=90, 
        leaf_font_size=11,
        ax=ax, 
        color_threshold=corte_visual, 
        above_threshold_color='#1a1a1a'
    )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Distancia', fontsize=11, labelpad=10)

    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0) # Cuadrícula sutil de fondo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show(block=False)
    
def plot_matrix_heatmap(matrix: pd.DataFrame, title: str, save_path: str = "matrix_heatmap.pdf"):
    set_latex_style()
    plt.figure(figsize=(8, 6))
    
    plt.pcolor(matrix, cmap='RdBu', vmin=-1, vmax=1)
    
    plt.colorbar()
    plt.yticks(np.arange(0.5, matrix.shape[0] + 0.5), matrix.index)
    plt.xticks(np.arange(0.5, matrix.shape[1] + 0.5), matrix.columns, rotation=90)
    plt.gca().invert_yaxis()
    
    plt.title(title, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show(block=False)

def plot_coincidence_heatmap(matrix: pd.DataFrame, save_path: str = "coincidence_heatmap.pdf"):
    set_latex_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap=sns.light_palette("#1f77b4", as_cmap=True),
                vmin=0, vmax=100, cbar_kws={"label": "Coincidencia (%)"}, square=True, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_survival_heatmap(all_stats_dict: dict, ticker_to_macrofamily: dict, save_path: str = "survival_heatmap.pdf"):
    set_latex_style()
    
    metrics_order = ['SR', 'SR_corr', 'MSR', 'Sortino', 'Kappa3', 'Omega', 'VaRR1', 'VaRR5', 'VaRR10', 'UPR', 'GR1', 'GR5']
    labels = {
        'VaRR1': r'VaRR$_{1\%}$', 'VaRR5': r'VaRR$_{5\%}$', 'VaRR10': r'VaRR$_{10\%}$', 
        'GR1': r'GR$_1$', 'GR5': r'GR$_5$', 'SR_corr': r'SR$_{\rho}$'
    }

    family_colors = {
        "Value": "#aec7e8",               
        "Growth": "#98df8a",              
        "Dividend": "#ffbb78",            
        "Low Volatility": "#c5b0d5",      
        "Defensive": "#ff9896",    
        "Fundamental": "#c49c94",    
        "ESG": "#f7b6d2",
        "Alternative": "#c7c7c7"
    }
    
    families_list = list(family_colors.keys())
    family_to_int = {fam: i for i, fam in enumerate(families_list)}
    
    cmap = ListedColormap([family_colors[fam] for fam in families_list])

    color_matrix = []
    annot_matrix = []

    for m in metrics_order:
        df_top5 = all_stats_dict[m].head(5)
        
        row_colors = []
        row_annots = []
        
        for _, row in df_top5.iterrows():
            ticker = row['Ticker']
            med_pos = row['Median_Position']
            
            family = ticker_to_macrofamily.get(ticker, "Alternative")
            row_colors.append(family_to_int[family])
            row_annots.append(f"{ticker}\n({med_pos:.1f}º)")
            
        color_matrix.append(row_colors)
        annot_matrix.append(row_annots)

    color_df = pd.DataFrame(
        color_matrix, 
        index=[labels.get(m, m) for m in metrics_order], 
        columns=['', '', '', '', ''] 
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        color_df, 
        annot=np.array(annot_matrix), 
        fmt="", 
        cmap=cmap, 
        vmin=0, vmax=len(families_list)-1, 
        linewidths=0.5, 
        linecolor='white',
        cbar=False, 
        annot_kws={"size": 11, "fontweight": "bold"}, 
        ax=ax
    )
    
    ax.tick_params(axis='x', length=0)

    legend_patches = [
        mpatches.Patch(color=color, label=fam.replace('_', ' ')) 
        for fam, color in family_colors.items()
    ]
    
    ax.legend(
        handles=legend_patches, 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0., 
        frameon=False, 
        title="Familias Smart Beta",
        title_fontproperties={'weight': 'bold', 'size': 11}
    )
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()
    return {}
    
def plot_dendrogram_heatmap(corr_matrix: pd.DataFrame, link: np.ndarray, title: str, save_path: str = "clustermap.pdf"):
    set_latex_style()
    
    order = sch.leaves_list(link)
    ordered_corr = corr_matrix.iloc[order, order]
    ordered_labels = ordered_corr.columns.tolist()
    
    fig = plt.figure(figsize=(10, 11))
    gs = plt.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.02)
    
    ax_dendro = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 1])

    sch.set_link_color_palette(['#1f4e79', '#a50026', '#404040', '#006d2c', '#542788'])
    max_d = link[-1, 2]
    sch.dendrogram(link, ax=ax_dendro, no_labels=True, color_threshold=0.7*max_d, 
                   above_threshold_color='#1a1a1a', link_color_func=None)
    ax_dendro.set_axis_off() 

    sns.heatmap(
        ordered_corr, 
        ax=ax_heatmap, 
        cbar_ax=ax_cbar, 
        cmap="RdBu", 
        vmin=-1, vmax=1, 
        annot=True, fmt=".2f", 
        annot_kws={"size": 8, "fontweight": "bold"},
        xticklabels=ordered_labels, 
        yticklabels=ordered_labels,
        linewidths=0.5, linecolor='white'
    )
    
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90)
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0)
    
    fig.suptitle(title, fontweight='bold', fontsize=14, y=0.95)
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show(block=False)