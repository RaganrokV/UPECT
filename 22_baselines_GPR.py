# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
from My_utils.evaluation_scheme import evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

#%%
"""
GPR 
Trip-level energy consumption prediction model for electric bus combining Markov-based 
speed profile generation and Gaussian processing regression
"""
#%%
"""local"""
with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

def labeling(Array):
    """label encoding"""
    # 对出行季节进行Label Encoding
    season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
    Array['出行季节'] = Array['出行季节'].map(season_mapping)

    # 对出行日期进行Label Encoding
    day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
                   'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
    Array['出行日期'] = Array['出行日期'].map(day_mapping)

    # 对出行时段进行Label Encoding
    period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
    Array['出行时段'] = Array['出行时段'].map(period_mapping)

    # 对车辆类型进行Label Encoding
    vehicle_mapping = {'Sedan': 0, 'SUV': 0.333, 'Sedan PHEV': 0.667, 'SUV PHEV': 1}
    Array['车辆类型'] = Array['车辆类型'].map(vehicle_mapping)

    Array['整备质量'] = Array['整备质量']/1880
    Array['电池能量'] = Array['电池能量'] /61.1
    Array['当前累积行驶里程'] = Array['当前累积行驶里程'] / 500000

    Array.loc[Array['VV'].apply(lambda x: isinstance(x, str)), 'VV'] = 0.1

    columns_to_drop = ["出行时间", "地点", "VIN"]
    Array = Array.drop(columns=columns_to_drop).astype(float)
    Array = Array.fillna(0)

    return Array

SUM_M = []
for i in [0.7,0.5,0.3,0.1]:
    M = []
    for arr in Test_vehicle:
        Array = labeling(arr)
        x = Array.values[:, 1:]
        y = Array.values[:, 0]

        X_train, X_test, y_train, y_test = \
            train_test_split(x, y, test_size=(1 - i), shuffle=False)

        # 添加判断，如果X_train只有一个样本，就复制自己变成两个
        if X_train.shape[0] == 1:
            X_train = np.vstack([X_train, X_train])
            y_train = np.append(y_train, y_train[0])

        # 定义内核和超参数网格
        param_grid = {
            'kernel': [RBF(length_scale=20.0), RBF(length_scale=10),
                        RBF(length_scale=5)],
            'alpha': [1e-6, 1e-3, 1e-1,1],  # 常数项（常量尺度）
            # 可以添加更多的超参数和取值范围
        }

        # 初始化高斯过程回归器
        gp_regressor = GaussianProcessRegressor(random_state=42,)

        # 创建网格搜索对象
        grid_search = GridSearchCV(estimator=gp_regressor, param_grid=param_grid, scoring='neg_mean_squared_error',
                                   cv=2)

        # 进行网格搜索
        grid_search.fit(X_train, y_train)

        # 获取最佳模型
        best_gp_regressor = grid_search.best_estimator_

        EC_Pred = best_gp_regressor.predict(X_test)

        EC_True = y_test

        Metric1 = np.array(evaluation(EC_True, EC_Pred))
        print("acc:", Metric1)
        M.append(Metric1)

    METRIC = np.vstack(M)
    SUM_M.append(METRIC)
#%%
"""Charge Car"""

with open("11-pretrained_incremental_learning/data/ChargeCar.pkl", "rb") as f:
    ChargeCar = pickle.load(f)


x = ChargeCar.values[:, 1:]
y = ChargeCar.values[:, 0]

X_train, X_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.9, shuffle=False)

# 定义内核和超参数网格
param_grid = {
    'kernel': [RBF(length_scale=20.0), RBF(length_scale=10),
                RBF(length_scale=5)],
     'alpha': [ 1e-6, 1e-3, 1e-1,1],  # 常数项（常量尺度）
    # 可以添加更多的超参数和取值范围
}

# 初始化高斯过程回归器
gp_regressor = GaussianProcessRegressor(random_state=42)

# 创建网格搜索对象
grid_search = GridSearchCV(estimator=gp_regressor, param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                            cv=5)

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_gp_regressor = grid_search.best_estimator_

EC_Pred = np.abs(best_gp_regressor.predict(X_test))

EC_True = y_test

Metric1 = np.array(evaluation(EC_True, EC_Pred))
print("acc:", Metric1)
#%%
"""BHD"""

with open("11-pretrained_incremental_learning/data/BHD.pkl", "rb") as f:
    BHD = pickle.load(f)

rename_dict = {
    'Date': '出行时间',
    'Distance [km]': '行程距离',
    'Duration [min]': '行程时间',
    'Battery State of Charge (Start)': '当前SOC',
    'Ambient Temperature (Start) [°C]': 'T'
}
BHD.rename(columns=rename_dict, inplace=True)
def labeling2(Array):
    """label encoding"""
    # 对出行季节进行Label Encoding
    season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
    Array['出行季节'] = Array['出行季节'].map(season_mapping)

    # 对出行日期进行Label Encoding
    day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
                   'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
    Array['出行日期'] = Array['出行日期'].map(day_mapping)

    # 对出行时段进行Label Encoding
    period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
    Array['出行时段'] = Array['出行时段'].map(period_mapping)

    # 对车辆类型进行Label Encoding
    vehicle_mapping = {'Sedan': 0, 'SUV': 0.333, 'Sedan PHEV': 0.667, 'SUV PHEV': 1}
    Array['车辆类型'] = Array['车辆类型'].map(vehicle_mapping)

    Array['整备质量'] = Array['整备质量']/1880
    Array['电池能量'] = Array['电池能量'] /61.1


    columns_to_drop = ["出行时间", "地点"]
    Array = Array.drop(columns=columns_to_drop).astype(float)
    Array = Array.fillna(0)

    return Array

BHD=labeling2(BHD)

BHD['当前SOC']=BHD['当前SOC']*100

x = BHD.values[:, 1:]
y = BHD.values[:, 0]

X_train, X_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.9, shuffle=False)

# 定义内核和超参数网格
param_grid = {
    'kernel': [RBF(length_scale=20.0), RBF(length_scale=10),
                RBF(length_scale=5)],
     'alpha': [ 1e-6, 1e-3, 1e-1,1],  # 常数项（常量尺度）
    # 可以添加更多的超参数和取值范围
}

# 初始化高斯过程回归器
gp_regressor = GaussianProcessRegressor(random_state=42)

# 创建网格搜索对象
grid_search = GridSearchCV(estimator=gp_regressor, param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                            cv=5)

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_gp_regressor = grid_search.best_estimator_

EC_Pred = np.abs(best_gp_regressor.predict(X_test))

EC_True = y_test

Metric1 = np.array(evaluation(EC_True, EC_Pred))
print("acc:", Metric1)
#%%
"""SpritMonitor"""

with open("11-pretrained_incremental_learning/data/SpritMonitor.pkl", "rb") as f:
    SpritMonitor = pickle.load(f)


rename_dict = {
    'quantity(kWh)': '行程能耗',
    'trip_distance(km)': '行程距离',
    'odometer': '当前累积行驶里程',
    'fuel_date': '出行时间',
    'avg_speed(km/h)': '行程平均速度',
}

def labeling(df):
    """label encoding"""
    # 对出行季节进行 Label Encoding
    season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
    df['出行季节'] = df['出行季节'].map(season_mapping)

    # 对出行日期进行 Label Encoding
    day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
                   'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
    df['出行日期'] = df['出行日期'].map(day_mapping)

    # 对出行时段进行 Label Encoding
    period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
    df['出行时段'] = df['出行时段'].map(period_mapping)

    # 对车辆类型进行 Label Encoding
    vehicle_mapping = {'Sedan': 0, 'SUV': 0.333, 'Sedan PHEV': 0.667, 'SUV PHEV': 1}
    df['车辆类型'] = df['车辆类型'].map(vehicle_mapping)

    # 对驾驶风格进行 Label Encoding
    style_mapping = {'Normal': 0, 'Moderate': 0.5, 'Fast': 1}
    df['driving_style'] = df['driving_style'].map(style_mapping)

    # 对轮胎类型进行 Label Encoding
    tire_mapping = {'Winter tires': 0, 'Summer tires': 1}
    df['tire_type'] = df['tire_type'].map(tire_mapping)

    df['整备质量'] = df['整备质量'] / 1880
    df['电池能量'] = df['电池能量'] / 61.1

    # 删除不需要的列
    columns_to_drop = ["出行时间", "地点"]
    df.drop(columns=columns_to_drop, inplace=True)

    # 将数据类型转换为 float，并填充缺失值为 0
    df = df.astype(float).fillna(0)

    return df

# 对每个 DataFrame 执行重命名和标签编码操作
SpritMonitor_labeled=[]
for df in SpritMonitor:
    df.rename(columns=rename_dict, inplace=True)  # 执行重命名操作
    SpritMonitor_labeled.append(labeling(df))  # 执行标签编码操作

for structured in SpritMonitor_labeled:

    x = structured.values[:, 1:]
    y = structured.values[:, 0]

    X_train, X_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.9, shuffle=False)

    # 定义内核和超参数网格
    param_grid = {
        'kernel': [RBF(length_scale=5.0), RBF(length_scale=1),
                   RBF(length_scale=0.5)],
        'alpha': [  1e-3, 1e-1,1,10],  # 常数项（常量尺度）
        # 可以添加更多的超参数和取值范围
    }

    # 初始化高斯过程回归器
    gp_regressor = GaussianProcessRegressor(random_state=42)

    # 创建网格搜索对象
    grid_search = GridSearchCV(estimator=gp_regressor, param_grid=param_grid,
                               scoring='neg_mean_squared_error',
                               cv=5)

    # 进行网格搜索
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_gp_regressor = grid_search.best_estimator_

    EC_Pred = np.abs(best_gp_regressor.predict(X_test))

    EC_True = y_test

    Metric1 = np.array(evaluation(EC_True, EC_Pred))
    print("acc:", Metric1)
#%%
"""VED"""

with open("11-pretrained_incremental_learning/data/VED.pkl", "rb") as f:
    VED = pickle.load(f)
VED=pd.concat(VED, ignore_index=True)
rename_dict = {
    '湿度': 'U',
    '可见度': 'VV',
    '风速': 'Ff',
    '温度': 'T',
    '降雨': 'RRR',
}

VED.rename(columns=rename_dict, inplace=True)  # 执行重命名操作

VED["VV"] = VED["VV"].replace("2.50V", "2.5")

def labeling(df):
    """label encoding"""
    # 对出行季节进行 Label Encoding
    season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
    df['出行季节'] = df['出行季节'].map(season_mapping)

    # 对出行日期进行 Label Encoding
    day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
                   'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
    df['出行日期'] = df['出行日期'].map(day_mapping)

    # 对出行时段进行 Label Encoding
    period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
    df['出行时段'] = df['出行时段'].map(period_mapping)

    # 对车辆类型进行 Label Encoding
    vehicle_mapping = {'Sedan': 0, 'SUV': 0.333, 'Sedan PHEV': 0.667, 'SUV PHEV': 1}
    df['车辆类型'] = df['车辆类型'].map(vehicle_mapping)

    df['整备质量'] = df['整备质量'] / 1880
    df['电池能量'] = df['电池能量'] / 61.1

    # 删除不需要的列
    columns_to_drop = ["出行时间", "地点"]
    df.drop(columns=columns_to_drop, inplace=True)

    # 将数据类型转换为 float，并填充缺失值为 0
    df = df.astype(float).fillna(0)

    return df


VED=labeling(VED) # 执行标签编码操作

x = VED.values[:, 1:]
y = VED.values[:, 0]

X_train, X_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.9, shuffle=False)

# 定义内核和超参数网格
param_grid = {
    'kernel': [RBF(length_scale=20.0), RBF(length_scale=10),
                RBF(length_scale=5)],
     'alpha': [ 1e-10, 1e-6, 1e-3, 1e-1],  # 常数项（常量尺度）
    # 可以添加更多的超参数和取值范围
}

# 初始化高斯过程回归器
gp_regressor = GaussianProcessRegressor(random_state=42)

# 创建网格搜索对象
grid_search = GridSearchCV(estimator=gp_regressor, param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                            cv=5)

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_gp_regressor = grid_search.best_estimator_

EC_Pred = np.abs(best_gp_regressor.predict(X_test))

EC_True = y_test

Metric1 = np.array(evaluation(EC_True, EC_Pred))
print("acc:", Metric1)

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(EC_True, 'ro-', label='True Values')  # 真实值用红点表示
plt.plot(EC_Pred, 'k*-', label='Predicted Values')  # 预测值用黑线表示
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('True Values vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

