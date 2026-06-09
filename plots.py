"""
PLOTS: VISUALIZACIÓN DE DATOS Y GRÁFICOS
Smart Beta ETF Universe - Quantitative Finance Master's Thesis
Autor: Alejandro Martinez
"""

import pandas as pd
import matplotlib.pyplot as plt
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

def plot_boxplots(df_data: pd.DataFrame, save_name: str = "boxplots_tfm.pdf"):
    """
    Genera un panel de tres boxplots para Volatilidad, Asimetría y Curtosis.
    """
    metrics_map = {
        'Volatility': 'Volatilidad', 
        'Skewness': 'Asimetría', 
        'Kurtosis': 'Curtosis'
    }

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    for ax, (col, title) in zip(axes, metrics_map.items()):
        sns.boxplot(
            y=df_data[col], ax=ax, color="white", width=0.25, linewidth=1.2,
            boxprops=dict(edgecolor="black", zorder=3),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            medianprops=dict(color="black", linewidth=1.8),
            flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black", alpha=0.5)
        )
        
        ax.set_title(title, fontweight='normal', pad=15)
        ax.set_xticks([])
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)

    plt.tight_layout(w_pad=4.0)
    plt.savefig(save_name, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()
    
def plot_market_model_regression(df_reg_sinc, df_res_alphas_betas, index_col, index_name, save_path):
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 11
    })
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x_market = df_reg_sinc[index_col].values
    x_grid = np.linspace(x_market.min(), x_market.max(), 100)
    
    for etf in df_res_alphas_betas['ETF']:
        ax.scatter(df_reg_sinc[index_col], df_reg_sinc[etf], color='#457B9D', alpha=0.01, s=1, rasterized=True)
    
    for _, row in df_res_alphas_betas.iterrows():
        alpha = row['Alpha']
        beta = row['Beta']
        ax.plot(x_grid, alpha + beta * x_grid, color='#1D3557', alpha=0.15, linewidth=0.5)
    
    mean_alpha = df_res_alphas_betas['Alpha'].mean()
    mean_beta = df_res_alphas_betas['Beta'].mean()
    label_average = (
        f'Ensemble Average Line ($\\bar{{\\alpha}}_{{\\mathrm{{ann}}}}$: {mean_alpha*252*100:.2f}\%, '
        f'$\\bar{{\\beta}}$: {mean_beta:.2f})')
    
    ax.plot(x_grid, mean_alpha + mean_beta * x_grid, color='#A72636', linestyle='-', linewidth=2, label=label_average)
    ax.plot(x_grid, 0 + 1 * x_grid, color='black', linestyle='--', linewidth=1.2, label='Market Neutral ($\\alpha = 0$, $\\beta = 1.0$)')
    ax.set_title(f'Market Model Regression: Smart Beta ETF Universe vs. {index_name}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(f'Benchmark Daily Return: {index_name} ($R_m$)', fontsize=11, labelpad=8)
    ax.set_ylabel(r'Smart Beta ETF Daily Return ($R_i$)', fontsize=11, labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.legend(loc='best', frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_cumulative_returns(diccionario_riqueza: dict):
    set_latex_style()
    metrics_map = {
    'SR': 'SR', 'SR_corr': 'SR$_{\\rho}$', 'MSR': 'MSR', 'VaRR1': 'VaRR(1%)', 
    'VaRR5': 'VaRR(5%)', 'VaRR10': 'VaRR(10%)', 'Omega': 'Omega', 
    'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': 'GR(1%)', 'GR5': 'GR(5%)'}
    
    elite_palette = [
        '#1D3557', '#2E5077', '#457B9D',
        '#2A9D8F', '#40798C', '#52796F',
        '#85182A', '#9B2226', '#A72636',
        '#8A5A44', '#B5838D', '#D4A373']
    
    color_dict = {key: elite_palette[i] for i, key in enumerate(metrics_map.keys())}
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 18), squeeze=False)
    axes = axes.flatten()

    for idx, (raw_title, values) in enumerate(diccionario_riqueza.items()):
        ax = axes[idx]
        rolling_bruto = values[0]
        rolling_neto = values[1]
        bh_bruto = values[2]
        bh_neto = values[3]
        metric_color = color_dict.get(raw_title, 'black')
        spread_bruta = rolling_bruto - bh_bruto
        spread_neta = rolling_neto - bh_neto
        ax.plot(spread_bruta, label='Spread (Sin Costes)', color='black', linestyle='--', linewidth=1.2)
        ax.plot(spread_neta, label='Spread (Con Costes)', color=metric_color, linestyle='-', linewidth=1.8)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.8)
        ax.set_title(metrics_map.get(raw_title, raw_title), fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, loc='best', frameon=False)

    plt.tight_layout(pad=4.0)
    plt.savefig("cumulative_spread_returns.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()


def plot_turnover_frequency(diccionario_frecuencia: dict, save_path: str = "turnover_frequency.pdf"):
    set_latex_style()
    
    metrics_map = {
    'SR': 'SR', 'SR_corr': 'SR$_{\\rho}$', 'MSR': 'MSR', 'VaRR1': 'VaRR(1%)', 
    'VaRR5': 'VaRR(5%)', 'VaRR10': 'VaRR(10%)', 'Omega': 'Omega', 
    'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': 'GR(1%)', 'GR5': 'GR(5%)'}
    
    elite_palette = [
        '#1D3557', '#2E5077', '#457B9D', 
        '#2A9D8F', '#40798C', '#52796F',
        '#85182A', '#9B2226', '#A72636', 
        '#8A5A44', '#B5838D', '#D4A373'  ]
    
    color_dict = {key: elite_palette[i] for i, key in enumerate(metrics_map.keys())}
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), sharey=True)
    axes = axes.flatten()

    for i, (col, title) in enumerate(metrics_map.items()):
        ax = axes[i]
        serie = diccionario_frecuencia[col]
        ax.bar(serie.index, serie.values, color=color_dict[col], edgecolor="black", linewidth=0.5, width=1.0)
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-0.5, 20.5)
        ax.grid(axis='y', linestyle=':', color='gray', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()
    return {}

def plot_survival_heatmap(all_stats_dict: dict, ticker_to_macrofamily: dict, save_path: str = "survival_heatmap.pdf"):
    
    set_latex_style()
    metrics_order = ['SR', 'SR_corr', 'MSR', 'Sortino', 'Kappa3', 'Omega', 'VaRR1', 'VaRR5', 'VaRR10', 'UPR', 'GR1', 'GR5']
    labels = {
        'VaRR1': r'VaRR(1%)', 'VaRR5': r'VaRR(5%)', 'VaRR10': r'VaRR(10%)', 
        'GR1': r'GR(1%)', 'GR5': r'GR(5%)', 'SR_corr': r'SR$_{\rho}$'}
    family_colors = {
        "Value": "#1D3557",               
        "Growth": "#457B9D",              
        "Dividend": "#2A9D8F",            
        "Low Volatility": "#52796F",      
        "Defensive": "#85182A",    
        "Fundamental": "#A72636",    
        "ESG": "#B5838D",
        "Alternative": "#D4A373"}
    
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
            tickers = row['Ticker']
            med_pos = row['Median_Position']
            family = ticker_to_macrofamily.get(tickers, "Alternative")
            row_colors.append(family_to_int[family])
            row_annots.append(f"{tickers}\n({med_pos:.1f}º)")
            
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
    'SR': 'SR', 'SR_corr': 'SR$_{\\rho}$', 'MSR': 'MSR', 'VaRR1': 'VaRR(1%)', 
    'VaRR5': 'VaRR(5%)', 'VaRR10': 'VaRR(10%)', 'Omega': 'Omega', 
    'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': 'GR(1%)', 'GR5': 'GR(5%)'
}
    
    elite_palette = [
        '#1D3557', '#2E5077', '#457B9D',
        '#2A9D8F', '#40798C', '#52796F',
        '#85182A', '#9B2226', '#A72636',
        '#8A5A44', '#B5838D', '#D4A373'
    ]
    
    color_dict = {key: elite_palette[i] for i, key in enumerate(metrics_map.keys())}
    rows, cols = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 18))
    axes = axes.flatten()

    for i, base_strategy in enumerate(perfiles):
        ax = axes[i]
        riqueza_base = diccionario_riqueza[benchmark][base_strategy]
        
        for j, ratio in enumerate(ratios):
            spread = diccionario_riqueza[ratio][base_strategy] - riqueza_base
            
            label_limpio = metrics_map.get(ratio, ratio)
            color_ratio = color_dict.get(ratio, '#000000') 
            ax.plot(spread.index, spread.values, linewidth=1.2, color=color_ratio, label=label_limpio)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'Weight Strategy: {base_strategy} (Spread w.r.t. {benchmark})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left', ncol=2, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig("selection_spreads.pdf", format='pdf', bbox_inches='tight', transparent=True)
    plt.show()
    return {}

def plot_allocation_spreads(diccionario_riqueza, fechas_oos, ratios, perfiles, benchmark='EW'):
    set_latex_style()
    
    metrics_map = {
    'SR': 'SR', 'SR_corr': 'SR$_{\\rho}$', 'MSR': 'MSR', 'VaRR1': 'VaRR(1%)', 
    'VaRR5': 'VaRR(5%)', 'VaRR10': 'VaRR(10%)', 'Omega': 'Omega', 
    'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': 'GR(1%)', 'GR5': 'GR(5%)'
}
    
    allocation_palette = {
        'EW': '#A9A9A9',   
        'VT': '#A8DADC',   
        'RRT': '#CCD5AE',  
        'ERC': '#D4A373',  
        'HRP': '#9D8189',  
        'HERC': '#2A9D8F', 
        'MVS': '#1D3557',  
        'GMV': '#85182A'   
    }
    
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    axes = axes.flatten()

    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        riqueza_base = diccionario_riqueza[ratio][benchmark]
        for strat in perfiles:
            if strat == benchmark:
                continue
            spread = diccionario_riqueza[ratio][strat] - riqueza_base
            ax.plot(spread.index, spread.values, linewidth=1.2, color=allocation_palette.get(strat, '#000000'), label=strat)
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

def plot_mhi(mhi_history, fechas_oos, ratios, perfiles):
    set_latex_style()
    
    metrics_map = {
    'SR': 'SR', 'SR_corr': 'SR$_{\\rho}$', 'MSR': 'MSR', 'VaRR1': 'VaRR(1%)', 
    'VaRR5': 'VaRR(5%)', 'VaRR10': 'VaRR(10%)', 'Omega': 'Omega', 
    'UPR': 'UPR', 'Kappa3': 'Kappa3', 'Sortino': 'Sortino', 'GR1': 'GR(1%)', 'GR5': 'GR(5%)'
}

    allocation_palette = {
        'EW': '#A9A9A9',   
        'VT': '#A8DADC',   
        'RRT': '#CCD5AE',  
        'ERC': '#D4A373',  
        'HRP': '#9D8189',  
        'HERC': '#2A9D8F', 
        'MVS': '#1D3557',  
        'GMV': '#85182A'   
    }
    
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    axes = axes.flatten()

    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        for perfil in perfiles:
            line_width = 1.8 if perfil in ['MVS', 'GMV', 'HERC'] else 1.0
            alpha_val = 1.0 if perfil in ['MVS', 'GMV', 'HERC'] else 0.6
            
            ax.plot(fechas_oos, mhi_history[ratio][perfil], linewidth=line_width, alpha=alpha_val, color=allocation_palette.get(perfil, '#000000'), label=perfil)

        titulo_ratio = metrics_map.get(ratio, ratio)
        ax.set_title(f'MHI - {titulo_ratio}', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc='upper left', ncol=2, frameon=False)
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