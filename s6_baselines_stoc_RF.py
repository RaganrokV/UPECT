#%%
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")

from My_utils.evaluation_scheme import evaluation

from sklearn.model_selection import GridSearchCV

#%%
"""
随机森林 （Stochastic RF） （Pengshun et al.引文2021）
Pengshun, L., Y. Zhang, K. Zhang, Y. Zhang, and K. Zhang. 2021. 
Prediction of electric bus energy consumption with stochastic speed profile generation modelling and data driven method based on real-world big data. 
Applied Energy 298:117204. 
doi:10.1016/j.apenergy.2021.117204.
"""
#%%
"""local"""
with open('/home/ps/haichao/11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

def labeling(Array):
    """label encoding"""
    # 对出行季节进行Label Encoding
    season_mapping = {'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}
    Array['出行季节'] = Array['出行季节'].map(season_mapping)

    # 对出行日期进行Label Encoding
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
                   'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    Array['出行日期'] = Array['出行日期'].map(day_mapping)

    # 对出行时段进行Label Encoding
    period_mapping = {'morning peak': 1, 'night peak': 2, 'other time': 3, "nighttime": 4}
    Array['出行时段'] = Array['出行时段'].map(period_mapping)

    # 对车辆类型进行Label Encoding
    vehicle_mapping = {'Sedan': 1, 'SUV': 2, 'Sedan PHEV': 3, 'SUV PHEV': 4}
    Array['车辆类型'] = Array['车辆类型'].map(vehicle_mapping)

    Array.loc[Array['VV'].apply(lambda x: isinstance(x, str)), 'VV'] = 0.1

    columns_to_drop = ["出行时间", "地点", "VIN"]
    Array = Array.drop(columns=columns_to_drop).astype(float)
    Array = Array.fillna(0)

    return Array

def generate_curve_beta(row):
    n = int(row['行程时间'] * 6)
    target_mean = row['行程平均速度']
    
    if n < 2:
        return pd.Series([np.nan]*4)
    
    # Beta分布参数设置
    a_accel = 2  # 加速阶段形状参数（左偏分布）
    b_accel = 5
    a_decel = 5  # 减速阶段形状参数（右偏分布）
    b_decel = 2
    
    # 生成加速阶段数据（前50%）
    split = n//2
    accel = np.random.beta(a_accel, b_accel, split)
    
    # 生成减速阶段数据（后50%）
    decel = np.random.beta(a_decel, b_decel, n-split)
    
    # 动态范围调整函数
    def scale_data(data, target):
        # 计算当前分布范围
        current_min, current_max = data.min(), data.max()
        # 扩展范围到实际速度区间（目标均值的50%-150%）
        scaled = data * (1.5*target - 0.5*target) + 0.5*target
        # 均值对齐
        scaled = scaled * (target / scaled.mean())
        return np.clip(scaled, 0.3*target, 1.7*target)  # 物理约束
    
    # 处理加速段
    accel_scaled = scale_data(accel, target_mean)
    
    # 添加加速度趋势（前10%数据增加上升趋势）
    ramp_up = np.linspace(0.8, 1.2, len(accel)//5)
    accel_scaled[:len(ramp_up)] *= ramp_up
    
    # 处理减速段
    decel_scaled = scale_data(decel, target_mean)
    
    # 添加减速趋势（后10%数据增加下降趋势）
    ramp_down = np.linspace(1.2, 0.8, len(decel)//5)
    decel_scaled[-len(ramp_down):] *= ramp_down
    
    # 合并完整曲线
    full_curve = np.concatenate([accel_scaled, decel_scaled])
    
    # 最终均值校准
    full_curve = full_curve * (target_mean / full_curve.mean())
    
    return pd.Series([
        full_curve[:split].mean(), full_curve[:split].var(),
        full_curve[split:].mean(), full_curve[split:].var()
    ])

# 应用新版生成函数
# stats_beta = df.apply(generate_curve_beta, axis=1)
#%%

SUM_M = []
P_all = []
# for p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
for p in [32]:
    M = []
    test_all_simu = []
    test_all_real = []
    for arr in Test_vehicle:
        # arr = Test_vehicle[5]
        generated=arr.apply(generate_curve_beta, axis=1)

        Array = labeling(arr)
        np.random.seed(42)

        Array=pd.concat([Array,generated], axis=1)
        Array = Array.reset_index(drop=True)
        x = Array.values[:, 1:]
        y = Array.values[:, 0]


        train_size = p
        test_size = int(len(x) * 0.3)
        # X_train, X_test, y_train, y_test = \
        #     train_test_split(x, y, test_size=(1 - i), shuffle=False)


        trainX = Array.iloc[:train_size, 1:].values
        trainY = Array.iloc[:train_size, 0].values

        testX = Array.iloc[-test_size:, 1:].values
        testY = Array.iloc[-test_size:, 0].values

        # 添加判断，如果X_train只有一个样本，就复制自己变成两个
        if trainX.shape[0] == 1:
            trainX = np.vstack([trainX, trainX])
            trainY = np.append(trainY, trainY[0])

        # 定义随机森林的参数网格
        param_grid = {
            'n_estimators': [5, 10, 20],
            'max_depth': [None, 1, 3, 5],
        }

        # 初始化随机森林回归器
        rf_regressor = RandomForestRegressor(random_state=42)

        # 在内部循环中进行网格搜索
        grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=2)
        grid_search.fit(trainX, trainY)

        # 使用最佳参数的模型进行预测
        best_rf_regressor = grid_search.best_estimator_
        EC_Pred = best_rf_regressor.predict(testX)

        EC_True = testY

        Metric1 = np.array(evaluation(EC_True, EC_Pred))
        print(Metric1)
        M.append(Metric1)

        test_all_simu.append(EC_Pred.reshape(-1, 1))
        test_all_real.append(EC_True.reshape(-1, 1))

        # break

    M2_test_all = np.array(evaluation(np.vstack(test_all_real),
                                      np.vstack(test_all_simu)))
    P_all.append(M2_test_all)

    METRIC = np.vstack(M)

    SUM_M.append(METRIC)

pd.DataFrame(SUM_M[-1])
#%%
data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/RF_data.pkl', 'wb') as f:
    pickle.dump(data, f)

#%% for scatter
indices_to_remove = [0, 15, 16, 17, 18]

# Iterate over the indices in reverse order to avoid index shifting issues while deleting
for index in sorted(indices_to_remove, reverse=True):
    del test_all_simu[index]
    del test_all_real[index]

EC_Pred = np.concatenate(test_all_simu, axis=0)
EC_True = np.concatenate(test_all_real, axis=0)

data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/RF_scatter.pkl', 'wb') as f:
    pickle.dump(data, f)
#%%
"""Charge Car"""

with open("11-pretrained_incremental_learning/data/ChargeCar.pkl", "rb") as f:
    ChargeCar = pickle.load(f)


x = ChargeCar.values[:, 1:]
y = ChargeCar.values[:, 0]

X_train, X_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.9, shuffle=False)

# 定义随机森林的参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
     'max_depth': [None, 10, 20, 30],
}

# 初始化随机森林回归器
rf_regressor = RandomForestRegressor(random_state=42)

# 在内部循环中进行网格搜索
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# 使用最佳参数的模型进行预测
best_rf_regressor = grid_search.best_estimator_
EC_Pred = best_rf_regressor.predict(X_test)

EC_True = y_test

Metric1 = np.array(evaluation(EC_True, EC_Pred))
print("acc:", Metric1)

data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('11-pretrained_incremental_learning/data_plot/RF_Chargecar.pkl', 'wb') as f:
    pickle.dump(data, f)
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

# 定义随机森林的参数网格
param_grid = {
    'n_estimators':[4,6,8],
     'max_depth': [None, 10, 20, 30],
}

# 初始化随机森林回归器
rf_regressor = RandomForestRegressor(random_state=42)

# 在内部循环中进行网格搜索
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# 使用最佳参数的模型进行预测
best_rf_regressor = grid_search.best_estimator_
EC_Pred = best_rf_regressor.predict(X_test)

EC_True = y_test

Metric1 = np.array(evaluation(EC_True, EC_Pred))
print("acc:", Metric1)

data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('11-pretrained_incremental_learning/data_plot/RF_BHD.pkl', 'wb') as f:
    pickle.dump(data, f)
#%%
"""SpritMonitor"""

with open("11-pretrained_incremental_learning/data/SpritMonitor.pkl", "rb") as f:
    SpritMonitor = pickle.load(f)

# 定义一个函数来删除满足条件的行
def filter_dataframes(dfs):
    for i in range(len(dfs)):
        # 计算 ratio
        ratio = dfs[i]['quantity(kWh)'] / dfs[i]['trip_distance(km)']

        # 删除大于 1 的行
        dfs[i] = dfs[i][(ratio <= 0.3) & (ratio >= 0.1)]

    return dfs


# 应用函数
SpritMonitor = filter_dataframes(SpritMonitor)

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

A=[]
B=[]
for structured in SpritMonitor_labeled:

    x = structured.values[:, 1:]
    y = structured.values[:, 0]

    X_train, X_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.9, shuffle=False)

    # 定义随机森林的参数网格
    param_grid = {
        'n_estimators':[50, 100, 200],
        'max_depth': [None, 10, 20, 30],
    }

    # 初始化随机森林回归器
    rf_regressor = RandomForestRegressor(random_state=42)

    # 在内部循环中进行网格搜索
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)

    # 使用最佳参数的模型进行预测
    best_rf_regressor = grid_search.best_estimator_
    EC_Pred = best_rf_regressor.predict(X_test)

    EC_True = y_test

    Metric1 = np.array(evaluation(EC_True, EC_Pred))
    print("acc:", Metric1)

    A.append(EC_True)
    B.append(EC_Pred)

EC_True=np.hstack(A).reshape(-1,1)
EC_Pred=np.hstack(B).reshape(-1,1)
Metric = np.array(evaluation(EC_True, EC_Pred))
print("acc:", Metric)

#%%
data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('11-pretrained_incremental_learning/data_plot/RF_SpritMonitor.pkl', 'wb') as f:
    pickle.dump(data, f)
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

# 定义随机森林的参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
     'max_depth': [None, 10, 20, 30],
}

# 初始化随机森林回归器
rf_regressor = RandomForestRegressor(random_state=42)

# 在内部循环中进行网格搜索
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# 使用最佳参数的模型进行预测
best_rf_regressor = grid_search.best_estimator_
EC_Pred = best_rf_regressor.predict(X_test)

EC_True = y_test

Metric1 = np.array(evaluation(EC_True, EC_Pred))
print("acc:", Metric1)

data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('11-pretrained_incremental_learning/data_plot/RF_VED.pkl', 'wb') as f:
    pickle.dump(data, f)