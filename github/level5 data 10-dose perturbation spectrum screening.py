import pandas as pd

# 路径
file_path = r"C:/桌面数据/selected_drug_expression_A549_MODZ稳定聚合.csv"

# 读取前4行作为列名索引
header_rows = pd.read_csv(file_path, nrows=4, header=None)
data = pd.read_csv(file_path, skiprows=4)

# 构造多级列名
multi_cols = pd.MultiIndex.from_arrays(header_rows.values)

# 替换列名
data.columns = multi_cols

# 找出Dose为10的列
dose_10_cols = [col for col in data.columns if str(col[3]).strip() == '10']

# 固定保留前4行（如DrugName等）
fixed_cols = [col for col in data.columns if col[0] == 'DrugName']

# 合并保留列
keep_cols = fixed_cols + dose_10_cols
data_filtered = data[keep_cols]

# 保存
output_path = r"C:/桌面数据/selected_dose_10_only_fixed.csv"
data_filtered.to_csv(output_path, index=False)

print(f"已正确筛选出 Dose = 10 的列并保存到：{output_path}")
