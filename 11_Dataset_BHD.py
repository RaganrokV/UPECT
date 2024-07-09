# -*- coding: utf-8 -*-
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")
#%%
file_path = r'E:\PHEV DATA\BHD\Overview.xlsx'
df = pd.read_excel(file_path)
#%%
BHD=df[['Date', 'Battery Temperature (Start) [°C]',
        'Battery State of Charge (Start)',
       'Unnamed: 8','Ambient Temperature (Start) [°C]',
       'Target Cabin Temperature', 'Distance [km]', 'Duration [min]',
       ]]

BHD.dropna(how='all', inplace=True)
BHD = BHD[BHD['Unnamed: 8'] >= 0]
BHD.reset_index(drop=True, inplace=True)
#https://ev-database.org/car/1004/BMW-i3-60-Ah
BHD['行程能耗'] =18.8*BHD['Unnamed: 8']
#ECR=1.63
#%%
# 添加出行季节列
BHD['Date'] = pd.to_datetime(BHD['Date'], format='%Y-%m-%d_%H-%M-%S')
BHD['出行季节'] = BHD['Date'].apply(lambda x: 'spring' if x.month in [3, 4, 5] else (
    'summer' if x.month in [6, 7, 8] else ('autumn' if x.month in [9, 10, 11] else 'winter')))

# 添加出行日期列
BHD['出行日期'] = BHD['Date'].apply(lambda x: x.strftime('%A'))

# 添加出行时段列
BHD['出行时段'] = BHD['Date'].apply(lambda x: 'morning peak' if 7 <= x.hour < 10 else (
    'night peak' if 15 <= x.hour < 18 else (
        'nighttime' if 22 <= x.hour <= 24 or 0 <= x.hour < 7 else 'other time')))

BHD['电池能量'] = 18.8
BHD['车辆类型'] = 'Sedan'
BHD['整备质量'] = 2029
BHD['地点'] = 'German'

BHD=BHD[['行程能耗','Date','Distance [km]', 'Duration [min]', 'Battery Temperature (Start) [°C]',
        'Battery State of Charge (Start)','Ambient Temperature (Start) [°C]',
        '出行季节','出行日期','出行时段','电池能量','车辆类型','整备质量',
             '地点','Target Cabin Temperature',]]

#%%
with open('11-pretrained_incremental_learning/data/BHD.pkl', 'wb') as file:
    pickle.dump(BHD, file)
