import pandas as pd
import numpy as np
import re

# ======== 1. 读取 CSV 数据 ========
file = r"F:\Data\HW\90th\Chw\urban_chw_diff.csv"
df = pd.read_csv(file, encoding="utf-8", sep=None, engine="python")

# 提取年份和 DOY
df["Year"] = df["Raster_File"].apply(lambda x: int(re.search(r"_(\d{4})TMAX_", x).group(1)))
df["DOY"] = df["Raster_File"].apply(lambda x: int(re.search(r"TMAX_(\d{3})", x).group(1)))

# 获取城市列
city_cols = [c for c in df.columns if str(c).isdigit()]

# ======== 2. 定义计算函数（新增 std_DOY） ========
def compute_hw_timing(sub):
    nonzero = sub[sub["value"] > 0]
    if len(nonzero) == 0:
        return pd.Series({
            "DOY_start": np.nan,
            "DOY_end": np.nan,
            "mean_DOY": np.nan,
            "std_DOY": np.nan
        })
    doy_start = nonzero["DOY"].min()
    doy_end   = nonzero["DOY"].max()
    mean_doy  = (nonzero["DOY"] * nonzero["value"]).sum() / nonzero["value"].sum()
    std_doy   = np.sqrt(((nonzero["DOY"] - mean_doy)**2 * nonzero["value"]).sum() / nonzero["value"].sum())
    # ↑ 加权标准差，如果你只想普通标准差可以改成：
    # std_doy = nonzero["DOY"].std()
    return pd.Series({
        "DOY_start": doy_start,
        "DOY_end": doy_end,
        "mean_DOY": mean_doy,
        "std_DOY": std_doy
    })

# ======== 3. 计算每个城市每年的结果 ========
results = []
for city in city_cols:
    tmp = df[["Year", "DOY", city]].rename(columns={city: "value"})
    grouped = tmp.groupby("Year").apply(compute_hw_timing)
    grouped["City_ID"] = city
    results.append(grouped.reset_index())

df_all = pd.concat(results, ignore_index=True)

# ======== 4. 转换为宽表并重新排列 ========
df_all_melt = df_all.melt(
    id_vars=["Year", "City_ID"],
    value_vars=["DOY_start", "DOY_end", "mean_DOY", "std_DOY"]
)
df_all_melt["Row_Label"] = (
    df_all_melt["Year"].astype(str) + "_" +
    df_all_melt["variable"].str.replace("mean_DOY", "DOY_mean").str.replace("std_DOY", "DOY_std")
)

final_df = df_all_melt.pivot_table(index="Row_Label", columns="City_ID", values="value").reset_index()
final_df.rename(columns={"Row_Label": "Raster_File"}, inplace=True)

# ======== 5. 分块排序 (start → end → mean → std)，插入空行 ========
def sort_key(label):
    year, metric = label.split("_DOY_")
    order = {"start": 1, "end": 2, "mean": 3, "std": 4}
    return (order.get(metric, 5), int(year))

final_df = final_df.sort_values(by="Raster_File", key=lambda x: x.map(sort_key)).reset_index(drop=True)

# ======== 6. 插入空行（每种类型之间） ========
blocks = []
for group in ["start", "end", "mean", "std"]:
    block = final_df[final_df["Raster_File"].str.contains(f"DOY_{group}")]
    blocks.append(block)
    blocks.append(pd.DataFrame([[""] * len(final_df.columns)], columns=final_df.columns))

final_with_blank = pd.concat(blocks, ignore_index=True)

# ======== 7. 输出结果 ========
output = r"F:\Data\HW\90th\Chw\urban_timing.xlsx"
final_with_blank.to_excel(output, index=False)
print(f"✅ 已输出分块式结果（含 std_DOY）：{output}")
