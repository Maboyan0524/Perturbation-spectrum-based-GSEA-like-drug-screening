import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm
import warnings
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', category=UserWarning, module='tkinter')

# 设置字体和样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 保留 sans-serif 以支持中文 fallback
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")
plt.rcParams['figure.dpi'] = 100


# === 1. 读取扰动谱数据：long format 转 wide format，并应用宽格式处理逻辑 ===
def load_expression_matrix_from_long_table(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df[["gene_id", "Treat_Sample", "LogFC"]].dropna()

        # 处理重复（取平均）
        duplicates = df.duplicated(subset=["gene_id", "Treat_Sample"], keep=False)
        if duplicates.any():
            print("⚠️ 检测到重复 gene_id-Treat_Sample 组合，自动对这些值取平均")
            df = df.groupby(["gene_id", "Treat_Sample"], as_index=False).mean()

        # Pivot to wide
        matrix = df.pivot(index="gene_id", columns="Treat_Sample", values="LogFC")
        matrix.columns.name = None

        # 应用宽格式额外处理（统一大写索引，确保数值，去除全NaN列）
        matrix.index = matrix.index.astype(str).str.upper()
        matrix = matrix.apply(pd.to_numeric, errors='coerce')
        matrix = matrix.dropna(axis=1, how='all')

        # 额外：替换任何inf值为NaN（防止下游计算问题）
        matrix = matrix.replace([np.inf, -np.inf], np.nan)

        print(f"✅ 加载表达矩阵: {matrix.shape[0]} 基因 × {matrix.shape[1]} 药物")
        return matrix

    except Exception as e:
        print(f"❌ 读取表达数据失败: {e}")
        return None


# === 2. 读取上下调基因集（添加检查逻辑） ===
def load_gene_set_from_symbol_files(up_path, down_path):
    try:
        up_genes = set(pd.read_csv(up_path)["gene_id"].astype(str).str.upper().dropna().tolist())
        down_genes = set(pd.read_csv(down_path)["gene_id"].astype(str).str.upper().dropna().tolist())

        print(f"✅ 上调基因集: {len(up_genes)} 个基因")
        print(f"✅ 下调基因集: {len(down_genes)} 个基因")

        # 检查基因集大小
        if len(up_genes) < 5:
            warnings.warn(f"⚠️ 上调基因集太小({len(up_genes)}个)，可能影响GSEA效力")
        if len(down_genes) < 5:
            warnings.warn(f"⚠️ 下调基因集太小({len(down_genes)}个)，可能影响GSEA效力")
        if len(up_genes) > 500:
            warnings.warn(f"⚠️ 上调基因集很大({len(up_genes)}个)，GSEA可能不够敏感")
        if len(down_genes) > 500:
            warnings.warn(f"⚠️ 下调基因集很大({len(down_genes)}个)，GSEA可能不够敏感")

        # 检查重叠
        overlap = up_genes & down_genes
        if overlap:
            warnings.warn(f"⚠️ 上下调基因集有 {len(overlap)} 个重叠基因，将从下调集中移除")
            down_genes -= overlap
            print(f"✅ 修正后下调基因集: {len(down_genes)} 个基因")

        return list(up_genes), list(down_genes)

    except Exception as e:
        print(f"❌ 读取基因集失败: {e}")
        return [], []


# === 3. 验证基因重叠情况 ===
def validate_gene_overlap(expr_matrix, up_genes, down_genes):
    expr_genes = set(expr_matrix.index)
    up_overlap = set(up_genes) & expr_genes
    down_overlap = set(down_genes) & expr_genes

    print(f"📊 基因重叠统计:")
    print(f"   上调基因在表达矩阵中: {len(up_overlap)}/{len(up_genes)} ({len(up_overlap) / len(up_genes) * 100:.1f}%)")
    print(
        f"   下调基因在表达矩阵中: {len(down_overlap)}/{len(down_genes)} ({len(down_overlap) / len(down_genes) * 100:.1f}%)")

    if len(up_overlap) == 0:
        raise ValueError("❌ 上调基因集与表达矩阵无重叠!")
    if len(down_overlap) == 0:
        raise ValueError("❌ 下调基因集与表达矩阵无重叠!")

    # 检查重叠比例
    if len(up_overlap) / len(up_genes) < 0.5:
        warnings.warn(f"⚠️ 上调基因集重叠率较低({len(up_overlap) / len(up_genes) * 100:.1f}%)，可能影响结果可靠性")
    if len(down_overlap) / len(down_genes) < 0.5:
        warnings.warn(f"⚠️ 下调基因集重叠率较低({len(down_overlap) / len(down_genes) * 100:.1f}%)，可能影响结果可靠性")

    return up_overlap, down_overlap


# === 4. 单次GSEA分数计算（使用实际表达值权重） ===
def compute_gsea_score_single(gene_expr_pairs, gene_set, weighted_score_type=1):
    N = len(gene_expr_pairs)
    if N == 0:
        return 0.0

    genes = [gene for gene, _ in gene_expr_pairs]
    expr_values = np.array([expr for _, expr in gene_expr_pairs])

    hits = np.array([gene in gene_set for gene in genes])
    Nh = hits.sum()

    if Nh == 0:
        return 0.0

    no_hits = ~hits

    if weighted_score_type == 0:
        Phit = np.cumsum(hits) / Nh
    else:
        weights = np.abs(expr_values)
        hit_weights = weights[hits]
        if hit_weights.sum() == 0:
            return 0.0
        Phit = np.cumsum(hits * weights) / hit_weights.sum()

    Pmiss = np.cumsum(no_hits) / (N - Nh)

    running_score = Phit - Pmiss
    es_pos = np.max(running_score)
    es_neg = -np.min(running_score)

    return es_pos if es_pos > es_neg else -es_neg


# === 5. 并行处理单个药物的分析（仅计算ES，无p值） ===
def analyze_single_drug(args):
    drug, expr_data, up_overlap, down_overlap = args

    expr = expr_data.dropna()
    if len(expr) == 0:
        # 如果无有效基因，返回NaN以便后续过滤
        return {
            "drug_id": drug,
            "es_up": np.nan,
            "es_down": np.nan,
            "wtcs": np.nan,
            "target_scores": {}
        }

    gene_expr_pairs = [(gene, val) for gene, val in expr.sort_values(ascending=False).items()]

    es_up = compute_gsea_score_single(gene_expr_pairs, up_overlap)
    es_down = compute_gsea_score_single(gene_expr_pairs, down_overlap)

    wtcs = (es_up - es_down) / 2

    target_scores = {}
    all_genes = list(up_overlap) + list(down_overlap)
    for gene in all_genes:
        target_scores[gene] = expr.get(gene, np.nan)

    return {
        "drug_id": drug,
        "es_up": es_up,
        "es_down": es_down,
        "wtcs": wtcs,
        "target_scores": target_scores
    }


# === 6. 主函数：执行GSEA + WTCS + Tau分析（取消统计检验） ===
def run_custom_wtcs(expr_matrix, up_genes, down_genes, n_processes=None):
    up_overlap, down_overlap = validate_gene_overlap(expr_matrix, up_genes, down_genes)

    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(expr_matrix.columns))

    print(f"🔄 开始计算WTCS分数...")
    print(f"   并行进程数: {n_processes}")

    args_list = [(drug, expr_matrix[drug], up_overlap, down_overlap) for drug in expr_matrix.columns]

    if n_processes > 1:
        with mp.Pool(processes=n_processes) as pool:
            results = list(tqdm(pool.imap(analyze_single_drug, args_list), total=len(args_list), desc="处理药物"))
    else:
        results = [analyze_single_drug(args) for args in tqdm(args_list, desc="处理药物")]

    print("✅ GSEA计算完成，开始排序和Tau计算...")

    df = pd.DataFrame(results)

    # 过滤掉WTCS为NaN的行（无效药物）
    initial_count = len(df)
    df = df.dropna(subset=['wtcs']).reset_index(drop=True)
    nan_count = initial_count - len(df)
    if nan_count > 0:
        print(f"⚠️ 过滤掉 {nan_count} 个无效药物（无有效WTCS分数）")

    if len(df) == 0:
        print("❌ 无有效WTCS分数，无法进行分析。")
        return pd.DataFrame()

    # 按WTCS排序 + rank
    df = df.sort_values(by="wtcs", ascending=True).reset_index(drop=True)
    df["rank"] = df["wtcs"].rank(method="min", ascending=True).astype(int)

    # Tau计算
    n = len(df)
    if n > 1:
        percentile_ranks = rankdata(df["wtcs"], method="average") / n
        tau_scores = (percentile_ranks * 200) - 100
    else:
        tau_scores = [0.0]
    df["tau"] = tau_scores

    # 效果分类（基于Tau，无显著性）
    def classify_effect(row):
        if row['tau'] < -90:
            return '强逆转'
        elif row['tau'] < -50:
            return '中等逆转'
        elif row['tau'] < 0:
            return '弱逆转'
        else:
            return '无效果'

    df['effect_category'] = df.apply(classify_effect, axis=1)

    # 靶点基因分数
    print("📊 添加靶点基因详细分数...")
    all_target_genes = list(up_overlap) + list(down_overlap)
    target_score_dict = {f"{gene}_score": [] for gene in all_target_genes}

    for _, row in df.iterrows():
        for gene in all_target_genes:
            target_score_dict[f"{gene}_score"].append(row["target_scores"].get(gene, np.nan))

    for gene in all_target_genes:
        df[f"{gene}_score"] = target_score_dict[f"{gene}_score"]

    # 按Tau排序（从低到高，即逆转潜力从强到弱）
    result_df = df.sort_values(by="tau", ascending=True).reset_index(drop=True)
    result_df['final_rank'] = range(1, len(result_df) + 1)

    print("✅ 分析完成!")

    # 摘要
    print(f"📈 结果摘要:")
    print(f"   药物总数: {len(result_df)}")
    print(f"   强逆转药物: {(result_df['effect_category'] == '强逆转').sum()} 个")
    print(f"   中等逆转药物: {(result_df['effect_category'] == '中等逆转').sum()} 个")
    print(f"   弱逆转药物: {(result_df['effect_category'] == '弱逆转').sum()} 个")
    print(f"   WTCS范围: {result_df['wtcs'].min():.4f} ~ {result_df['wtcs'].max():.4f}")
    print(f"   Tau范围: {result_df['tau'].min():.1f} ~ {result_df['tau'].max():.1f}")

    return result_df


# === 新增: ES散点图绘制函数 ===
def plot_es_scatter(results_df, figsize=(12, 10)):
    """绘制ES_up vs ES_down散点图"""
    # 确保DataFrame已按WTCS升序排序（低WTCS为好，即逆转效果强）
    sorted_df = results_df.sort_values(by='wtcs', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    # 根据WTCS值设置颜色
    scatter = ax.scatter(sorted_df['es_up'], sorted_df['es_down'],
                         c=sorted_df['wtcs'], cmap=LinearSegmentedColormap.from_list("custom", ['#FF9A9B', '#47B2FF']),
                         alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

    # 添加对角线
    min_es = min(sorted_df['es_up'].min(), sorted_df['es_down'].min())
    max_es = max(sorted_df['es_up'].max(), sorted_df['es_down'].max())
    ax.plot([min_es, max_es], [min_es, max_es], color='black', linestyle='--', linewidth=2, alpha=1.0,
            label='ES_up = ES_down')

    # 设置标签和标题
    ax.set_xlabel('上调基因富集分数 (ES_up)')
    ax.set_ylabel('下调基因富集分数 (ES_down)')
    ax.set_title('上调 vs 下调基因富集分数散点图')

    # 添加颜色条并美化
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, aspect=30)
    cbar.set_label('WTCS分数', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(1.5)

    ax.legend()
    plt.tight_layout()

    # 保存图像到指定路径，600 DPI，并添加bbox_inches='tight'以优化裁剪
    plt.savefig(r"C:\Users\19834\Desktop\文章\补充材料\ES_scatter_plot.png", dpi=600, bbox_inches='tight')
    plt.show()


# === 新增: Tau vs z-score 图绘制函数 ===
def plot_tau_vs_zscore(figsize=(10, 6)):
    """绘制Tau vs z-score理论映射图"""
    fig, ax = plt.subplots(figsize=figsize)

    # 生成Tau值
    tau = np.linspace(-100, 100, 1000)
    percentile = (tau + 100) / 200
    # 避免inf
    percentile = np.clip(percentile, 1e-5, 1 - 1e-5)
    z = norm.ppf(percentile)
    z = np.clip(z, -4, 4)

    # 使用LineCollection创建渐变粗线
    points = np.array([tau, z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm_col = plt.Normalize(z.min(), z.max())
    cmap = LinearSegmentedColormap.from_list("custom", ['#D8AEC8', '#47B2FF'])
    lc = LineCollection(segments, cmap=cmap, norm=norm_col)
    lc.set_array(z)
    lc.set_linewidth(3)  # 增加线宽
    line = ax.add_collection(lc)

    # 添加关键点
    key_z = np.array([-3.0, -1.5, 0, 1.5, 3.0])
    key_percentile = norm.cdf(key_z)
    key_tau = (key_percentile * 200) - 100
    ax.scatter(key_tau, key_z, c=key_z, cmap=cmap, norm=norm_col, s=100, edgecolor='black', zorder=3)

    # 添加颜色条并美化
    cbar = plt.colorbar(line, ax=ax, orientation='vertical', fraction=0.046, pad=0.1, aspect=30)
    cbar.set_label('z-score', rotation=270, labelpad=15)
    cbar.set_ticks(key_z)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(1.5)

    # 设置标签和标题
    ax.set_xlabel('Tau Score (percentile)')
    ax.set_ylabel('z-score')
    ax.set_title('Drug Reversal Potential (Tau vs z-score)')

    # 添加水平线和网格
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 调整轴范围
    ax.set_xlim(-100, 100)
    ax.set_ylim(-4, 4)

    plt.tight_layout()

    # 保存图像到指定路径，600 DPI，并添加bbox_inches='tight'以优化裁剪
    plt.savefig(r"C:\Users\19834\Desktop\文章\补充材料\Tau_vs_zscore_plot.png", dpi=600, bbox_inches='tight')
    plt.show()



# === 7. 主程序入口 ===
if __name__ == "__main__":
    # 替换为你的实际路径
    long_table_path = "D:/朱浩宇ITCM表达谱比对数据/results.csv"
    up_gene_path = "D:/朱浩宇ITCM表达谱比对数据/上调.csv"
    down_gene_path = "D:/朱浩宇ITCM表达谱比对数据/下调.csv"

    try:
        print("📥 加载表达数据...")
        expr_matrix = load_expression_matrix_from_long_table(long_table_path)

        if expr_matrix is None:
            raise ValueError("数据加载失败，请检查文件格式。")

        print("\n📥 加载基因集...")
        up_genes, down_genes = load_gene_set_from_symbol_files(up_gene_path, down_gene_path)

        if not up_genes or not down_genes:
            raise ValueError("基因集加载失败，请检查文件格式。")

        print("\n🔬 开始WTCS + Tau分析（无统计检验）...")

        results_df = run_custom_wtcs(
            expr_matrix, up_genes, down_genes,
            n_processes=4  # 可以调整
        )

        # 显示Top结果
        print(f"\n🏆 Top 10 药物逆转排名（Tau 从低到高）:")
        display_cols = ["drug_id", "wtcs", "tau", "effect_category"]
        print(results_df[display_cols].head(10).to_string(index=False))

        # 保存
        output_path = "D:/朱浩宇ITCM表达谱比对数据/WTCS_TAU_results_no_stats.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ 结果已保存至: {output_path}")

        # 新增: 绘制ES散点图
        print("\n📊 绘制ES散点图...")
        plot_es_scatter(results_df)

        # 新增: 绘制Tau vs z-score图
        print("\n📊 绘制Tau vs z-score图...")
        plot_tau_vs_zscore()

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback

        print("详细错误信息:")
        print(traceback.format_exc())