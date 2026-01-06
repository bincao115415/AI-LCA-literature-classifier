import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 基础绘图设置，偏学术杂志风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

INPUT_FILE = Path('savedrecs_full_classified.xlsx')
TREND_PNG = Path('ai_lca_trends.png')
TREND_SVG = Path('ai_lca_trends.svg')
STACKED_PNG = Path('ai_lca_group_stacked.png')
STACKED_SVG = Path('ai_lca_group_stacked.svg')
HEATMAP_PNG = Path('ai_lca_group_heatmap.png')
HEATMAP_SVG = Path('ai_lca_group_heatmap.svg')
GROUP_EXCEL = Path('ai_lca_group_counts.xlsx')
COMBINED_PNG = Path('ai_lca_trend_heatmap_combined.tiff')
COMBINED_SVG = Path('ai_lca_trend_heatmap_combined.svg')
DPI = 600
YEARS_FORCE_ZERO = [2004, 2007]
FIG_WIDTH_IN = 183 / 25.4  # 183 mm Nature 双栏宽度
TREND_HEIGHT_IN = 3.2   # 高度保留，宽度加倍以减少拥挤
STACKED_HEIGHT_IN = 3.4
HEATMAP_HEIGHT_IN = 3.2
ANNOT_FONTSIZE = 9 
SUBFIG_LABEL_SIZE = 8

GROUP_MAP = {
    'Group 1': 'G1: Goal, Scope & Inventory',
    'Group 2': 'G2: Characterization & Emission Factor Prediction',
    'Group 3': 'G3: Streamlined/Surrogate LCA',
    'Group 4': 'G4: Optimization, Uncertainty & Interpretation',
    'Group 5': 'G5: Dynamic LCA & Advanced Paradigms',
}
GROUP_ORDER = list(GROUP_MAP.values())
# 颜色：更偏 Nature 期刊的克制配色
GROUP_COLORS = ['#4C6A91', '#6C9A8B', '#C97C5D', '#7D5BA6', '#9C6B55']
HEATMAP_CMAP = sns.color_palette("mako_r", as_cmap=True)
TREND_BAR_COLOR = '#1c8041'   # 绿色
TREND_LINE_COLOR = '#e55709'  # 橙色


def parse_groups(cell):
    """Parse group cell to a list; tolerate strings like "['Group 3']"."""
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        stripped = cell.strip()
        if not stripped or stripped == '[]':
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [stripped]
    return []


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if 'is_ai_lca' not in df.columns:
        raise KeyError("'is_ai_lca' column not found.")

    # 只保留判定为 True 的样本，兼容字符串 true/false
    mask = df['is_ai_lca'].astype(str).str.lower().isin(['true', '1', 'yes'])
    df = df.loc[mask].copy()

    if 'Publication Year' not in df.columns:
        raise KeyError("'Publication Year' column not found.")
    df = df.dropna(subset=['Publication Year'])
    df['Publication Year'] = df['Publication Year'].astype(int)

    if 'groups' in df.columns:
        df['groups_list'] = df['groups'].apply(parse_groups)
    else:
        df['groups_list'] = [[] for _ in range(len(df))]
    return df


def build_group_counts(df: pd.DataFrame):
    exploded = df.explode('groups_list')
    exploded['groups_list'] = exploded['groups_list'].fillna('')
    exploded = exploded[exploded['groups_list'] != '']
    exploded['Group_Label'] = exploded['groups_list'].map(GROUP_MAP).fillna(exploded['groups_list'])

    counts = (
        exploded.groupby(['Publication Year', 'Group_Label'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=GROUP_ORDER, fill_value=0)
        .sort_index()
    )
    counts = counts.reindex(sorted(counts.index.union(YEARS_FORCE_ZERO)), fill_value=0)
    counts.loc[YEARS_FORCE_ZERO] = 0
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100
    return counts, shares


def plot_trend(yearly_counts: pd.Series):
    if yearly_counts.empty:
        print('No data to plot annual trend.')
        return

    cumulative = yearly_counts.cumsum()
    years = yearly_counts.index

    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH_IN, TREND_HEIGHT_IN))
    bars = ax1.bar(
        years,
        yearly_counts.values,
        color=TREND_BAR_COLOR,
        alpha=0.82,
        width=0.65,
        label='Annual publications',
    )

    ax2 = ax1.twinx()
    ax2.plot(
        years,
        cumulative.values,
        color=TREND_LINE_COLOR,
        marker='o',
        linewidth=1.1,
        markersize=4,
        label='Cumulative publications',
    )

    for bar, year in zip(bars, years):
        height = bar.get_height()
        if height > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.4,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=ANNOT_FONTSIZE,
            )

    ax1.set_ylabel('Annual publications')
    ax1.set_xlabel('Publication year')
    ax2.set_ylabel('Cumulative publications')
    # 为避免 x 轴拥挤，必要时稀疏年份刻度（双栏宽度容纳更多刻度）
    if len(years) > 20:
        tick_years = years[::2]
    else:
        tick_years = years
    ax1.set_xticks(tick_years)
    ax1.set_xticklabels(tick_years, rotation=45)
    # 固定刻度：左轴 0-140 每 20，右轴与之等距映射为 0-? 每 50
    left_step = 20
    right_step = 50
    left_base_max = 140
    ratio = right_step / left_step  # 2.5，确保网格线等距
    right_min = -25

    # 确保覆盖实际数据：若累计值超出映射，需要提升左轴上限（按 20 递增）
    left_needed_for_line = np.ceil(cumulative.max() / ratio / left_step) * left_step if cumulative.max() > 0 else 0
    left_max = max(left_base_max, yearly_counts.max(), left_needed_for_line)
    # 左轴上限向上取 20 的倍数
    left_max = np.ceil(left_max / left_step) * left_step
    right_range = left_max * ratio
    right_max = right_min + right_range

    left_ticks = np.arange(0, left_max + left_step, left_step)
    right_ticks = np.arange(right_min, right_max + right_step, right_step)

    ax1.set_ylim(0, left_max)
    ax2.set_ylim(right_min, right_max)
    ax1.set_yticks(left_ticks)
    ax2.set_yticks(right_ticks)
    ax1.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.6)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)

    fig.tight_layout()
    fig.savefig(TREND_PNG, dpi=DPI)
    fig.savefig(TREND_SVG, dpi=DPI, format='svg')
    plt.close(fig)


def plot_group_stacked(counts: pd.DataFrame):
    if counts.empty:
        print('No group data to plot stacked chart.')
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, STACKED_HEIGHT_IN))
    counts.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=GROUP_COLORS[: len(counts.columns)],
        width=0.82,
        edgecolor='white',
    )

    for container in ax.containers:
        labels = [int(v.get_height()) if v.get_height() >= 5 else '' for v in container]
        ax.bar_label(container, labels=labels, label_type='center', fontsize=ANNOT_FONTSIZE, color='black')

    ax.set_ylabel('Occurrences of functional domains')
    ax.set_xlabel('Publication year')
    if len(counts.index) > 20:
        tick_positions = list(range(0, len(counts.index), 2))
        tick_years = counts.index[::2]
    else:
        tick_positions = list(range(len(counts.index)))
        tick_years = counts.index
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_years, rotation=45)
    ax.legend(
        title='The five functional domains of AI-empowered LCA',
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor='white',
        edgecolor='white',
        framealpha=1.0,
        fancybox=False,
    )
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(STACKED_PNG, dpi=DPI)
    fig.savefig(STACKED_SVG, dpi=DPI, format='svg')
    plt.close(fig)


def plot_group_heatmap(counts: pd.DataFrame):
    if counts.empty:
        print('No group data to plot heatmap.')
        return

    # 适度降低高度，避免图像显得过高
    plt.figure(figsize=(FIG_WIDTH_IN, HEATMAP_HEIGHT_IN))
    hm = sns.heatmap(
        counts.T,
        annot=True,
        fmt='d',
        cmap=HEATMAP_CMAP,
        annot_kws={'fontsize': ANNOT_FONTSIZE},
        cbar_kws={'label': 'Occurrences'},
        linewidths=0.5,
    )
    hm.collections[0].colorbar.set_label('Occurrences', fontsize=ANNOT_FONTSIZE)
    hm.collections[0].colorbar.ax.tick_params(labelsize=ANNOT_FONTSIZE)
    if len(counts.index) > 20:
        tick_positions = list(range(0, len(counts.index), 2))
        tick_years = counts.index[::2]
    else:
        tick_positions = list(range(len(counts.index)))
        tick_years = counts.index
    plt.xlabel('Publication year')
    plt.ylabel('The five functional domains of AI-empowered LCA')
    plt.xticks(ticks=tick_positions, labels=tick_years, rotation=45)
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=DPI)
    plt.savefig(HEATMAP_SVG, dpi=DPI, format='svg')
    plt.close()


def plot_combined_trend_heatmap(yearly_counts: pd.Series, group_counts: pd.DataFrame):
    if yearly_counts.empty or group_counts.empty:
        print('No data to plot combined figure.')
        return

    years = yearly_counts.index.tolist()
    n_years = len(years)
    positions = np.arange(n_years)  # cell edges at i..i+1
    centers = positions + 0.5
    width = 1.0  # match heatmap cell width (full width)

    cumulative = yearly_counts.cumsum()

    # 宽度比例：左边绘图区占大头，右边留一小列给 Colorbar
    # 为了让 Colorbar 的宽度接近一个年份的柱子宽度
    # 主绘图区域有 n_years 个柱子，所以宽度比例大概设为 n_years : 1 (或 1.2 以留出边距)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, TREND_HEIGHT_IN + HEATMAP_HEIGHT_IN + 1.0)) # 增加高度给下方文字
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[TREND_HEIGHT_IN, HEATMAP_HEIGHT_IN],
        width_ratios=[n_years, 0.8],  # 动态设置比例
        hspace=0.14,  # 增加垂直间距，给垂直的年份标签留出足够空间，使其位于两图中间
        wspace=0.05,  # 水平间距
    )

    # 上方趋势图（图 a），放在 [0, 0]
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(
        positions,
        yearly_counts.values,
        color=TREND_BAR_COLOR,
        alpha=0.82,
        width=width,
        align='edge',
        label='Annual publications',
        linewidth=0.5,
        edgecolor='white',
    )
    ax2 = ax1.twinx()
    ax2.plot(
        centers,
        cumulative.values,
        color=TREND_LINE_COLOR,
        marker='o',
        linewidth=1.1,
        markersize=4,
        label='Cumulative publications',
    )

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.4,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=ANNOT_FONTSIZE,
            )

    ax1.set_ylabel('Annual publications')
    ax2.set_ylabel('Cumulative publications')
    ax1.set_xlim(0, n_years)

    # 固定刻度：左轴 0-140 每 20，右轴与之等距映射为 0-? 每 50
    left_step = 20
    right_step = 50
    left_base_max = 140
    ratio = right_step / left_step
    right_min = -25

    left_needed_for_line = np.ceil(cumulative.max() / ratio / left_step) * left_step if cumulative.max() > 0 else 0
    left_max = max(left_base_max, yearly_counts.max(), left_needed_for_line)
    left_max = np.ceil(left_max / left_step) * left_step
    right_range = left_max * ratio
    right_max = right_min + right_range

    left_ticks = np.arange(0, left_max + left_step, left_step)
    right_ticks = np.arange(right_min, right_max + right_step, right_step)

    ax1.set_ylim(0, left_max)
    ax2.set_ylim(right_min, right_max)
    ax1.set_yticks(left_ticks)
    ax2.set_yticks(right_ticks)
    ax1.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.6)
    ax2.grid(False)

    # x 轴刻度：完全隐藏趋势图的 X 轴标签和刻度
    ax1.set_xticks(centers)
    ax1.set_xticklabels([])
    ax1.tick_params(axis='x', which='both', length=0, labelbottom=False) 

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False, fontsize=ANNOT_FONTSIZE)
    # ax1.text(-0.08, 1.02, '(a)', transform=ax1.transAxes, fontsize=SUBFIG_LABEL_SIZE, fontweight='bold', va='bottom')

    # 下方热力矩阵（图 b），放在 [1, 0]，共享 x 定位
    ax_hm = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_cbar = fig.add_subplot(gs[1, 1])
    
    heat_data = group_counts.reindex(years).T

    hm = sns.heatmap(
        heat_data,
        ax=ax_hm,
        cbar_ax=ax_cbar,
        annot=True,
        fmt='d',
        cmap=HEATMAP_CMAP,
        annot_kws={'fontsize': ANNOT_FONTSIZE},
        cbar_kws={'label': 'Occurrences'},
        linewidths=0.5,
    )
    hm.collections[0].colorbar.set_label('Occurrences', fontsize=ANNOT_FONTSIZE)
    hm.collections[0].colorbar.ax.tick_params(labelsize=ANNOT_FONTSIZE)

    # 将年份标签放在热力图顶部 (top)，即两图之间
    # 旋转 90 度 (垂直)
    # 设置对齐方式使其位于两图中间
    tick_positions = centers
    tick_labels = years
    
    ax_hm.set_xticks(tick_positions)
    # ha='left' 配合 rotation=90，文本从下往上延伸，起点对齐刻度
    # pad=5 将起点向上推离热力图边缘
    ax_hm.set_xticklabels(tick_labels, rotation=90, va='center', ha='left', rotation_mode='anchor')
    ax_hm.xaxis.tick_top()
    ax_hm.xaxis.set_label_position('top')
    ax_hm.set_xlabel('')  # 去掉 publication year
    ax_hm.tick_params(axis='x', pad=5, length=0) # length=0 去掉刻度线，只留文字

    # 简化 Y 轴标签为 G1-G5，并保持水平
    ax_hm.set_yticklabels(['G1', 'G2', 'G3', 'G4', 'G5'], rotation=0)
    ax_hm.set_ylabel('The five functional domains of AI-empowered LCA')
    ax_hm.set_xlim(0, n_years)
    
    # 统一左侧 Y 轴标签的位置，使其垂直对齐
    # 使用 set_label_coords 固定横坐标（axes coordinates，负数表示在轴左侧）
    # 0.5 表示垂直居中
    LABEL_PAD_X = -0.05  # 调整此值以拉近距离 (原为 -0.09)
    ax1.yaxis.set_label_coords(LABEL_PAD_X, 0.5)
    ax_hm.yaxis.set_label_coords(LABEL_PAD_X, 0.5)

    # 在底部添加文字说明 G1-G5 的具体名称
    # 获取全称，并只保留冒号后面的部分
    # 使用 ax_hm.text 确保相对于热力图定位
    desc_text = []
    for k, v in GROUP_MAP.items():
        short_k = k.replace('Group ', 'G')
        full_name = v.split(': ')[1]
        desc_text.append(f"{short_k}: {full_name}")
    
    # 将说明分为两行显示
    if len(desc_text) >= 3:
        line1 = "; ".join(desc_text[:3])
        line2 = "; ".join(desc_text[3:])
        final_desc = f"{line1}\n{line2}"
    else:
        final_desc = "; ".join(desc_text)

    # 相对于 ax_hm 底部定位 (y < 0)
    # 调整 y 值以控制距离热力图底部的距离
    ax_hm.text(
        0.0, -0.05, 
        final_desc, 
        transform=ax_hm.transAxes,
        fontsize=8,  # 单独控制字号，比全局 ANNOT_FONTSIZE (9) 小
        ha='left', 
        va='top',
        linespacing=1.5
    )

    # 调整布局
    # 增加 bottom 给底部文字 (因为文字虽然绑定在 ax 上，但仍需图窗空间)
    # left 稍微减小，因为 Y 轴标签拉近了
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.15)
    
    fig.savefig(COMBINED_PNG, dpi=DPI)
    fig.savefig(COMBINED_SVG, dpi=DPI, format='svg')
    # 也可以同时保存 PNG 方便预览
    fig.savefig(str(COMBINED_PNG).replace('.tiff', '.png'), dpi=DPI)
    plt.close(fig)


def main():
    if not INPUT_FILE.exists():
        print(f'未找到文件 {INPUT_FILE}, 请检查路径。')
        return

    df = load_data(INPUT_FILE)
    if df.empty:
        print('筛选后没有 AI-LCA 数据，无法绘图。')
        return

    yearly_counts = df.groupby('Publication Year').size().sort_index()
    yearly_counts = yearly_counts.reindex(sorted(yearly_counts.index.union(YEARS_FORCE_ZERO)), fill_value=0)
    yearly_counts.loc[YEARS_FORCE_ZERO] = 0
    plot_trend(yearly_counts)

    group_counts, group_shares = build_group_counts(df)
    with pd.ExcelWriter(GROUP_EXCEL) as writer:
        group_counts.to_excel(writer, sheet_name='absolute_counts')
        group_shares.round(2).to_excel(writer, sheet_name='share_percent')

    plot_group_stacked(group_counts)
    plot_group_heatmap(group_counts)
    plot_combined_trend_heatmap(yearly_counts, group_counts)

    print('输出完成：')
    print(f'- 年度趋势：{TREND_PNG} 和 {TREND_SVG}')
    print(f'- 研究类别堆叠图：{STACKED_PNG} 和 {STACKED_SVG}')
    print(f'- 研究类别热力图：{HEATMAP_PNG} 和 {HEATMAP_SVG}')
    print(f'- 类别统计 Excel：{GROUP_EXCEL}')
    print(f'- 组合图（趋势+热力）：{COMBINED_PNG} 和 {COMBINED_SVG}')


if __name__ == '__main__':
    main()
