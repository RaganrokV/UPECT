# -*- coding: utf-8 -*-
import pandas as pd
import pickle
#%%

imiev = pd.read_csv(r'E:\PHEV DATA\SpritMonitor-Crawler\mitsubishi_imiev.csv', encoding='latin-1')
golfev = pd.read_csv(r'E:\PHEV DATA\SpritMonitor-Crawler\volkswagen_e_golf.csv', encoding='latin-1')

#%%
imiev_ECR=imiev["consumption(kWh/100km)"].mean()
imiev=imiev[['fuel_date','odometer', 'trip_distance(km)',
             'tire_type', 'city', 'motor_way', 'country_roads',
             'driving_style', 'quantity(kWh)']]

golfev_ECR=golfev["consumption(kWh/100km)"].mean()
golfev=golfev[['fuel_date','trip_distance(km)', 'quantity(kWh)',
               'tire_type', 'city','motor_way', 'country_roads',
               'driving_style',  'avg_speed(km/h)']]
#%%
# 添加出行季节列
imiev['fuel_date'] = pd.to_datetime(imiev['fuel_date'], format='%d.%m.%Y')
imiev['出行季节'] = imiev['fuel_date'].apply(lambda x: 'spring' if x.month in [3, 4, 5] else (
    'summer' if x.month in [6, 7, 8] else ('autumn' if x.month in [9, 10, 11] else 'winter')))

# 添加出行日期列
imiev['出行日期'] = imiev['fuel_date'].apply(lambda x: x.strftime('%A'))

# 添加出行时段列
imiev['出行时段'] = imiev['fuel_date'].apply(lambda x: 'morning peak' if 7 <= x.hour < 10 else (
    'night peak' if 15 <= x.hour < 18 else (
        'nighttime' if 22 <= x.hour <= 24 or 0 <= x.hour < 7 else 'other time')))

imiev['电池能量'] = 16
imiev['车辆类型'] = 'Sedan'
imiev['整备质量'] = 1080
imiev['地点'] = 'German'

imiev=imiev[['quantity(kWh)','fuel_date', 'trip_distance(km)','odometer',
             '出行季节','出行日期','出行时段','电池能量','车辆类型','整备质量',
             '地点','tire_type', 'city', 'motor_way', 'country_roads','driving_style',]]
imiev['quantity(kWh)'].fillna(imiev['trip_distance(km)'] * imiev_ECR/100, inplace=True)
#%%
golfev['fuel_date'] = pd.to_datetime(golfev['fuel_date'], format='%d.%m.%Y')
golfev['出行季节'] = golfev['fuel_date'].apply(lambda x: 'spring' if x.month in [3, 4, 5] else (
    'summer' if x.month in [6, 7, 8] else ('autumn' if x.month in [9, 10, 11] else 'winter')))

# 添加出行日期列
golfev['出行日期'] =golfev['fuel_date'].apply(lambda x: x.strftime('%A'))

# 添加出行时段列
golfev['出行时段'] = golfev['fuel_date'].apply(lambda x: 'morning peak' if 7 <= x.hour < 10 else (
    'night peak' if 15 <= x.hour < 18 else (
        'nighttime' if 22 <= x.hour <= 24 or 0 <= x.hour < 7 else 'other time')))

golfev['电池能量'] = 35.8
golfev['车辆类型'] = 'Sedan'
golfev['整备质量'] = 1573
golfev['地点'] = 'German'
#ECR=13.6
golfev=golfev[['quantity(kWh)','fuel_date', 'trip_distance(km)','avg_speed(km/h)',
             '出行季节','出行日期','出行时段','电池能量','车辆类型','整备质量',
             '地点','tire_type', 'city', 'motor_way', 'country_roads','driving_style',]]

golfev['trip_distance(km)'] = pd.to_numeric(golfev['trip_distance(km)'], errors='coerce')
golfev['avg_speed(km/h)'].fillna(golfev['avg_speed(km/h)'].mean(), inplace=True)
golfev['trip_distance(km)'].fillna(golfev['quantity(kWh)'] * 100/golfev_ECR, inplace=True)
golfev['quantity(kWh)'].fillna(golfev['trip_distance(km)'] * 13.6/100, inplace=True)
#%%

SpritMonitor=[imiev,golfev]
with open('11-pretrained_incremental_learning/data/SpritMonitor.pkl', 'wb') as file:
    pickle.dump(SpritMonitor, file)