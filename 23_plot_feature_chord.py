# -*- coding: utf-8 -*-
import pickle
import warnings
from mpl_chord_diagram import chord_diagram

from scipy.interpolate import lagrange

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib import rcParams


#%%
with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

with open('11-pretrained_incremental_learning/data/Train_vehicle.pkl', 'rb') as file:
    Train_vehicle = pickle.load(file)
#%%

Array=pd.concat(Train_vehicle+Test_vehicle, ignore_index=True)


"""label encoding"""
# 对出行季节进行Label Encoding
season_mapping = {'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}
Array['出行季节'] = Array['出行季节'].map(season_mapping)

# 对出行日期进行Label Encoding
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
               'Friday': 5, 'Saturday': 6, 'Sunday': 7}
Array['出行日期'] = Array['出行日期'].map(day_mapping)

# 对出行时段进行Label Encoding
period_mapping = {'morning peak': 1, 'night peak': 2, 'other time': 3,"nighttime":4}
Array['出行时段'] = Array['出行时段'].map(period_mapping)

# 对车辆类型进行Label Encoding
vehicle_mapping = {'Sedan': 1, 'SUV': 2, 'Sedan PHEV': 3, 'SUV PHEV': 4}
Array['车辆类型'] = Array['车辆类型'].map(vehicle_mapping)

Array.loc[Array['VV'].apply(lambda x: isinstance(x, str)), 'VV'] = 0.1

columns_to_drop = ["出行时间", "地点", "VIN"]
Array = Array.drop(columns=columns_to_drop).astype(float)
Array = Array.fillna(0)
#%%

rename_dict = {
    '行程能耗': "Energy consumption",
    '行程时间': "Duration",
    '行程距离': "Distance",
    '行程平均速度': "Speed",
    '当前SOC': "State of charge",
    '当前累积行驶里程': "Odometer readings",
    '单体电池电压极差': "Battery voltage range",
    '单体电池温度极差': "Battery temperature range",
    '绝缘电阻值': "Insulation resistance",
    'T': "Temperature",
    'Po': "Pressure",
    'U': "Humidity",
    'Ff': "Wind speed",
    'VV': "Visibility",
    'RRR': "Precipitation",
    '出行季节': "Season",
    '出行日期': "Date",
    '出行时段': "Period",
    '电池能量': "Battery capacity",
    '车辆类型': "Vehicle Type",
    '整备质量': "Curb Weight"
}

Array.rename(columns=rename_dict, inplace=True)

Array = Array[['Energy consumption', 'Duration', 'Distance', 'Speed', 'State of charge', 'Odometer readings',
               'Battery voltage range', 'Battery temperature range', 'Insulation resistance', 'Pressure', 'Humidity',
               'Wind speed', 'Visibility', 'Temperature', 'Precipitation', 'Season', 'Date', 'Period', 'Battery capacity', 'Vehicle Type', 'Curb Weight']]

plt.rcParams['font.family'] = 'Times New Roman'


correlation_matrix = Array.corr()

C = chord_diagram(mat=correlation_matrix, names=Array.columns,
                  rotate_names=90,alpha=.7)

plt.savefig(r"11-pretrained_incremental_learning/Fig/feature_chord.svg", dpi=600)

plt.show()
#%%
# from pandas.plotting import scatter_matrix
# scatter_matrix(Array, alpha=0.2, figsize=(15, 15), diagonal='kde')
# plt.show()