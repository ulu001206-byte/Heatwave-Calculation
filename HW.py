import pandas as pd
import numpy as np
import geopandas as gpd
import os
import re


def extract_heatwave_events(data_column):
    """提取连续≥3天的热浪事件列表"""
    events = []
    current_event = []

    for value in data_column:
        if value > 0:
            current_event.append(value)
        else:
            if len(current_event) >= 3:
                events.append(current_event)
            current_event = []
    if len(current_event) >= 3:
        events.append(current_event)

    return events


def calculate_basic_indicators(data_column):
    """计算基本热浪指标：HWT, HWF, HWD, HWC"""
    heatwave_events = extract_heatwave_events(data_column)
    hwt = len(heatwave_events)  # 热浪事件个数
    hwf = sum(len(event) for event in heatwave_events)  # 热浪日总数
    hwd = hwf / hwt if hwt > 0 else 0  # 平均持续时间
    hwc = sum(sum(event) for event in heatwave_events)  # 累积热量
    return {'HWT': hwt, 'HWF': hwf, 'HWD': hwd, 'HWC': hwc}


def calculate_hwi_hw(data_column, basic_indicators=None):
    """计算HWI和HW"""
    if basic_indicators is None:
        basic_indicators = calculate_basic_indicators(data_column)
    heatwave_events = extract_heatwave_events(data_column)
    event_intensities = [np.mean(event) for event in heatwave_events]
    hwi = np.mean(event_intensities) if event_intensities else 0
    hw = basic_indicators['HWF'] * hwi * basic_indicators['HWD'] if hwi > 0 else 0
    return {'HWI': hwi, 'HW': hw}


def process_csv_and_generate_shapefiles(input_csv, input_shp, output_csv, output_folder):
    df = pd.read_csv(input_csv)
    if "Raster_File" not in df.columns:
        raise ValueError("CSV 文件的第一列必须是 'Raster_File'")

    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: max(x, 0) if isinstance(x, (int, float)) else x)
    data = df.iloc[:, 1:].to_numpy()

    for col in range(data.shape[1]):
        nonzero_indices = np.where(data[:, col] > 0)[0]
        if len(nonzero_indices) >= 3:
            valid_mask = np.zeros_like(data[:, col], dtype=bool)
            for i in range(len(nonzero_indices) - 2):
                if nonzero_indices[i + 2] - nonzero_indices[i] == 2:
                    valid_mask[nonzero_indices[i]:nonzero_indices[i + 2] + 1] = True
            data[:, col] = np.where(valid_mask, data[:, col], 0)
        else:
            data[:, col] = 0

    df.iloc[:, 1:] = data
    indicators_df = pd.DataFrame(columns=df.columns)

    year_groups = {}
    for index, row in df.iterrows():
        if row["Raster_File"] == "Mean":
            continue
        match = re.search(r"_(\d{4})TMIN", row["Raster_File"])
        if match:
            year = match.group(1)
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(row.iloc[1:])
    annual_indicators = {}
    for year, rows in year_groups.items():
        year_df = pd.DataFrame(rows)
        year_indicators = {
            f'{year}_HWT': [], f'{year}_HWF': [], f'{year}_HWD': [],
            f'{year}_HWC': [], f'{year}_HWI': [], f'{year}_HW': []
        }
        for col in year_df.columns:
            basic = calculate_basic_indicators(year_df[col])
            year_indicators[f'{year}_HWT'].append(basic['HWT'])
            year_indicators[f'{year}_HWF'].append(basic['HWF'])
            year_indicators[f'{year}_HWD'].append(basic['HWD'])
            year_indicators[f'{year}_HWC'].append(basic['HWC'])
            hwi_hw = calculate_hwi_hw(year_df[col], basic_indicators=basic)
            year_indicators[f'{year}_HWI'].append(hwi_hw['HWI'])
            year_indicators[f'{year}_HW'].append(hwi_hw['HW'])

        annual_indicators[year] = year_indicators
        for indicator_name, values in year_indicators.items():
            indicators_df.loc[len(indicators_df)] = [indicator_name] + values

    def calculate_global_indicators(annual_indicators, num_locations):
        """计算全局指标（多年平均）"""
        global_indicators = {k: [] for k in ['HWT', 'HWF', 'HWD', 'HWC', 'HWI', 'HW']}

        # 对每个空间位置分别计算多年平均
        for location_idx in range(num_locations):
            for key in global_indicators:
                yearly_values = []
                for year in sorted(annual_indicators.keys()):  # 按年份排序
                    year_key = f"{year}_{key}"
                    # 确保该位置有数据
                    if (year_key in annual_indicators[year] and
                            location_idx < len(annual_indicators[year][year_key])):
                        value = annual_indicators[year][year_key][location_idx]
                        # 只添加有效数值（避免NaN或无效值）
                        if not np.isnan(value) and value is not None:
                            yearly_values.append(value)

                # 计算该位置该指标的多年平均值
                if yearly_values:
                    global_indicators[key].append(np.mean(yearly_values))
                else:
                    global_indicators[key].append(0)  # 如果没有数据，设为0

        return global_indicators

    # 获取空间位置数量（CSV列数-1，因为第一列是Raster_File）
    num_locations = len(df.columns) - 1

    # 使用新的函数计算全局指标
    global_indicators = calculate_global_indicators(annual_indicators, num_locations)

    for key, values in global_indicators.items():
        indicators_df.loc[len(indicators_df)] = [key] + list(values)

    final_df = pd.concat([df, indicators_df], ignore_index=True)
    final_df.to_csv(output_csv, index=False)

    gdf = gpd.read_file(input_shp)
    from shapely.geometry import Point
    new_geoms = []
    for geom in gdf.geometry:
        if hasattr(geom, 'x') and hasattr(geom, 'y'):
            x, y = geom.x, geom.y
            # 简单限制范围
            x = max(min(x, 180), -180)
            y = max(min(y, 90), -90)
            new_geoms.append(Point(x, y))
        else:
            new_geoms.append(geom)

    gdf.geometry = new_geoms

    total_gdf = gdf.copy()
    for key in global_indicators:
        total_gdf[key] = 0
    for fid in total_gdf.index:
        col = str(fid)
        if col in final_df.columns[1:]:
            for key in global_indicators:
                val_row = final_df[final_df["Raster_File"] == key]
                if not val_row.empty:
                    total_gdf.loc[fid, key] = val_row[col].values[0]
    total_gdf = total_gdf.set_crs('EPSG:4326')
    total_gdf.to_file(os.path.join(output_folder, "Total.shp"))

    for year in year_groups:
        year_gdf = gdf.copy()
        for key in ['HWT', 'HWF', 'HWD', 'HWC', 'HWI', 'HW']:
            year_gdf[f"{year}_{key}"] = 0
        for fid in year_gdf.index:
            col = str(fid)
            if col in final_df.columns[1:]:
                for key in ['HWT', 'HWF', 'HWD', 'HWC', 'HWI', 'HW']:
                    row = final_df[final_df["Raster_File"] == f"{year}_{key}"]
                    if not row.empty:
                        year_gdf.loc[fid, f"{year}_{key}"] = row[col].values[0]
        year_gdf.to_file(os.path.join(output_folder, f"{year}.shp"), driver="ESRI Shapefile")

    print(f"处理完成，CSV 保存为 {output_csv}，Shapefiles 存放于 {output_folder}")


# 使用示例路径（请确保这些路径存在）
input_csv = r"F:\Data\HW\90th\Nhw\urban_nhw_diff.csv"
input_shp = r"F:\Data\HW\point\city_cluster.shp"
output_csv = r"F:\Data\HW\90th\Nhw\urban_nhw1.csv"
output_folder = r"F:\Data\HW\90th\Nhw\urban"

process_csv_and_generate_shapefiles(input_csv, input_shp, output_csv, output_folder)
