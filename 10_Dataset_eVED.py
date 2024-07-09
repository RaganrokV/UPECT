# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings
import pickle
warnings.filterwarnings("ignore")
#%%
df = pd.read_csv(r'E:\PHEV DATA\VED\0_VED_orig_data.csv')
#%%
VED=df[['datetime', 'vehid', 'trip','timestamp_ms','oat_degc','speed_kmh',
         'lat', 'lon','airconditioning_w', 'heaterpower_w', 'hvbattery_a',
       'hvbattery_soc_per', 'hvbattery_v', 'hourlyprecipitation',
       'hourlyrelativehumidity','hourlyvisibility', 'hourlywinddirection',
        'hourlywindspeed']]
#%%
VED['hourlyprecipitation'] = VED['hourlyprecipitation'].replace('T', 0)
VED['hourlyprecipitation'] = VED['hourlyprecipitation'].replace(np.nan, 0)
VED['hourlywinddirection'] = VED['hourlywinddirection'].replace('VRB', 0)
columns_to_interpolate = ['hourlyrelativehumidity', 'hourlyvisibility',
                          'hourlywinddirection', 'hourlywindspeed']
VED[columns_to_interpolate] = VED[columns_to_interpolate].\
    interpolate(method='linear', axis=0)

VED[columns_to_interpolate] = VED[columns_to_interpolate].fillna(method='ffill')
#%%
# 创建一个空列表用于存储拆分后的数据框
split_data = []

# 使用 groupby 方法拆分数据并存储在列表中
for vehid, group_data in VED.groupby('vehid'):
    split_data.append(group_data)

#%%
def calculate_distance(latitudes, longitudes):
    # 将经纬度转换为弧度
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)

    # 计算相邻坐标点之间的纬度和经度差
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)

    # 使用 Haversine 公式计算距离
    R = 6371000  # 地球半径（单位：米）
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[:-1].values) * np.cos(lat_rad[1:].values) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = R * c

    return distances

#%%
DATA=[]
for data in split_data:


    DF = pd.DataFrame(columns=['行程能耗','出行时间', '行程时间', '行程距离',
                               '行程平均速度','当前SOC'])

    for trip, trip_df in data.groupby('trip'):

        sum_current = trip_df['hvbattery_a']
        sum_voltage = trip_df['hvbattery_v']
        t=trip_df['timestamp_ms'].diff().fillna(0)/1000
        energy_consumption = np.abs(((sum_current * sum_voltage) * t / 1000 / 3600).sum())

        trip_start_time = trip_df['datetime'].iloc[0]


        trip_avg_speed = trip_df['speed_kmh'].mean()

        distances = calculate_distance(trip_df['lat'], trip_df['lon'])
        trip_distance = np.sum(distances)/ 1000


        trip_duration = (trip_df['timestamp_ms'].iloc[-1] -
                         trip_df['timestamp_ms'].iloc[0]) / 60/1000

        current_soc = trip_df['hvbattery_soc_per'].iloc[0]

        oat_degc= trip_df['oat_degc'].iloc[0]
        airconditioning_w = trip_df['airconditioning_w'].iloc[0]
        heaterpower_w = trip_df['heaterpower_w'].iloc[0]
        hourlyprecipitation= trip_df['hourlyprecipitation'].iloc[0]
        hourlyrelativehumidity = trip_df['hourlyrelativehumidity'].iloc[0]
        hourlyvisibility = trip_df['hourlyvisibility'].iloc[0]
        hourlywinddirection = trip_df['hourlywinddirection'].iloc[0]
        hourlywindspeed = trip_df['hourlywindspeed'].iloc[0]


        feature = {
            '行程能耗': energy_consumption,
            '出行时间':trip_start_time,
            '行程时间': trip_duration,
            '行程距离': trip_distance,
            '行程平均速度': trip_avg_speed,
            '当前SOC':current_soc,
            "温度":oat_degc,
            "空调功率": airconditioning_w,
            "加热器功率":heaterpower_w,
            "降雨":hourlyprecipitation,
            "湿度":hourlyrelativehumidity,
            "可见度":hourlyvisibility,
            "风向": hourlywinddirection,
            "风速": hourlywindspeed,
        }
        DF = DF.append(feature, ignore_index=True)

        DF['出行时间'] = pd.to_datetime(DF['出行时间'], format="%Y-%m-%d %H:%M:%S")
        DF['出行季节'] = DF['出行时间'].apply(lambda x: 'spring' if x.month in [3, 4, 5] else (
            'summer' if x.month in [6, 7, 8] else ('autumn' if x.month in [9, 10, 11] else 'winter')))

        # 添加出行日期列
        DF['出行日期'] = DF['出行时间'].apply(lambda x: x.strftime('%A'))

        # 添加出行时段列
        DF['出行时段'] = DF['出行时间'].apply(lambda x: 'morning peak' if 7 <= x.hour < 10 else (
            'night peak' if 15 <= x.hour < 18 else (
                'nighttime' if 22 <= x.hour <= 24 or 0 <= x.hour < 7 else 'other time')))

        DF['电池能量'] = 24
        DF['车辆类型'] = 'Sedan'
        DF['整备质量'] = 1493
        DF['地点'] = 'US'

    DATA.append(DF)

#%%
with open('11-pretrained_incremental_learning/data/VED.pkl', 'wb') as file:
    pickle.dump(DATA, file)





