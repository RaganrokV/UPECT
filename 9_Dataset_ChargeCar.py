# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import numpy as np
import os
#%%
# 指定文件夹路径
folder_path = r'E:\PHEV DATA\Charge_Car'

# 用于存储所有txt文件数据的列表
data_list = []

# 遍历文件夹中的所有文件和子文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 检查文件扩展名是否为txt
        if file.endswith('.txt'):
            # 拼接文件路径
            file_path = os.path.join(root, file)
            # 使用read_csv()函数读取txt文件，并添加到数据列表中
            data = pd.read_csv(file_path, delimiter=',', header=None, names=[
                "GMT time", "relative time (s)", "elevation (m)", "planar distance (m)",
                "adjusted distance (m)", "speed (m/s)", "acceleration (m/s^2)",
                "power based on model (kW)", "actual power (kW)", "current (amps)",
                "voltage (V)"
            ])
            # data['GMT time'] = pd.to_datetime(data['GMT time'])
            data_list.append(data)

#%%
# 将时间偏移量转换为时间间隔的函数
def convert_to_normal_time(time_offset):
    base_time = pd.Timestamp('2011-09-01')
    normal_time = base_time + pd.to_timedelta(time_offset, unit='s')
    return normal_time


# 对每个DataFrame的GMT time列执行转换操作
for df in data_list:
    df['GMT time'] = df['GMT time'].apply(convert_to_normal_time)
#%%
# 存储处理后的数据框的列表
processed_data_list = []

# 循环遍历每个数据框
for df in data_list:
    # 找到相邻值之间差距超过1000的索引
    split_indices = df[df['relative time (s)'].diff() > 1000].index.tolist()

    # 如果没有需要拆分的索引，直接将当前数据框添加到列表中
    if not split_indices:
        processed_data_list.append(df)
    else:
        # 添加第一个拆分点之前的部分
        processed_data_list.append(df.iloc[:split_indices[0]])

        # 循环处理连续的拆分点
        for i in range(len(split_indices) - 1):
            start_index = split_indices[i] + 1
            end_index = split_indices[i + 1]
            processed_data_list.append(df.iloc[start_index:end_index])

        # 添加最后一个拆分点之后的部分
        processed_data_list.append(df.iloc[split_indices[-1] + 1:])

#%%
DF = pd.DataFrame(columns=['行程能耗', '行程时间', '行程距离', '行程平均速度'])
for trip_df in processed_data_list:

    sum_current = trip_df['current (amps)']
    sum_voltage = trip_df['voltage (V)']
    energy_consumption =  np.abs(((sum_current * sum_voltage) * 2 / 1000 / 3600).sum())



    trip_avg_speed = trip_df['speed (m/s)'].mean()*3.6

    trip_distance = trip_df['adjusted distance (m)'].sum()/1000

    trip_duration = (trip_df['relative time (s)'].iloc[-1]-
                     trip_df['relative time (s)'].iloc[0]) / 60

    feature = {
        '行程能耗': energy_consumption,
        '行程时间': trip_duration,
        '行程距离': trip_distance,
        '行程平均速度': trip_avg_speed,
    }
    DF = DF.append(feature, ignore_index=True)

#%%
ChargeCar=DF
with open('11-pretrained_incremental_learning/data/ChargeCar.pkl', 'wb') as file:
    pickle.dump(ChargeCar, file)
