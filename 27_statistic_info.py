# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
#%%
with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

with open('11-pretrained_incremental_learning/data/Train_vehicle.pkl', 'rb') as file:
    Train_vehicle = pickle.load(file)
#%%

Array=pd.concat(Train_vehicle+Test_vehicle, ignore_index=True)

#%%
vehicle_counts = {}

# 遍历 Test_vehicle 中的每个 DataFrame
for df in Train_vehicle:
    # 检查 DataFrame 的长度是否小于 10
    if len(df) < 10:
        continue

    # 获取当前 DataFrame 中的车辆类型
    vehicle_type = df["车辆类型"].iloc[0]  # 假设车辆类型在该列的第一个值是唯一的

    # 将车辆类型添加到字典中，如果已存在，则数量加1
    if vehicle_type in vehicle_counts:
        vehicle_counts[vehicle_type] += 1
    else:
        vehicle_counts[vehicle_type] = 1

# 打印每种车辆类型及其数量
for vehicle_type, count in vehicle_counts.items():
    print(f"{vehicle_type}: {count}")

#%%
# 计算最大值、最小值、均值和方差
#、小、微型出租客运汽车使用8年（行驶里程60万公里）
Array.loc[Array['VV'].apply(lambda x: isinstance(x, str)), 'VV'] = 0.1
filtered_array = Array[(Array['单体电池电压极差'] <= 4) &
                      (Array['单体电池温度极差'] <= 10) &
                      (Array['当前累积行驶里程'] < 500000) &
                      (Array['当前累积行驶里程'] >= 5) &
                      (Array['绝缘电阻值'] >= 1000) &
                      (Array['行程距离'] <= 501)]

max_values =filtered_array.max()
min_values = filtered_array.min()
mean_values = filtered_array.mean()
variance_values = filtered_array.std()

# 创建一个新的数据框来汇总统计信息
summary_df = pd.DataFrame({
    'max': max_values,
    'min': min_values,
    'mean': mean_values,
    'variance': variance_values
})

print(summary_df)
#%%

