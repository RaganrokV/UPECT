# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
import xgboost as xgb
from My_utils.evaluation_scheme import evaluation

from sklearn.model_selection import GridSearchCV
#%%
"""
XGBoost （Zhang et al.引文2020）
Zhang, J., Z. Wang, P. Liu, and Z. Zhang. 2020. 
Energy consumption analysis and prediction of electric vehicles based on real-world driving data. 
Applied Energy 275:115408. 
doi:10.1016/j.apenergy.2020.115408.
"""
#%%
with open('/home/ps/haichao/11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

with open('/home/ps/haichao/11-pretrained_incremental_learning/normalization_params.pkl', 'rb') as f:
    min_values, max_values = pickle.load(f)


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

import numpy as np
import pandas as pd
from scipy.stats import norm

def mcmc_speed_generator(target_mean, n_samples, burn_in=10, max_accel=3):
    """
    MCMC速度序列生成器
    参数：
    target_mean - 目标平均速度（km/h）
    n_samples   - 需要生成的样本数
    burn_in     - 预烧期迭代次数
    max_accel   - 最大加速度约束（km/h/s）
    """
    # 物理约束参数
    dt = 10  # 采样间隔秒数（每10秒一个点）
    max_speed_change = max_accel * dt  # 每步最大速度变化
    
    # 初始化链
    speeds = [target_mean]
    current_speed = target_mean
    
    # 
    proposal_std = target_mean * 0.15  # 
    
    # MCMC迭代
    for _ in range(burn_in + n_samples - 1):
        # 生成候选样本（基于运动学约束）
        proposal = current_speed + np.random.normal(0, proposal_std)
        
        # 物理约束条件
        speed_change = abs(proposal - current_speed)
        if speed_change > max_speed_change:
            proposal = current_speed + np.sign(proposal - current_speed) * max_speed_change
        
        # Metropolis-Hastings接受概率（这里使用简单对称建议分布）
        # 目标分布：以全局均值为中心的高斯分布
        target_current = norm.pdf(current_speed, target_mean, proposal_std*2)
        target_proposal = norm.pdf(proposal, target_mean, proposal_std*2)
        
        acceptance = min(1, target_proposal / target_current)
        
        # 接受/拒绝
        if np.random.rand() < acceptance:
            current_speed = proposal
        
        # 仅保留预烧期后的样本
        if _ >= burn_in:
            speeds.append(current_speed)
    
    # 均值校准
    speeds = np.array(speeds)
    speeds = speeds * (target_mean / np.mean(speeds))
    
    return speeds

# 集成到数据处理流程
def generate_curve_mcmc(row):
    n = int(row['行程时间'] * 6)
    target_mean = row['行程平均速度']
    
    if n < 2:
        return pd.Series([np.nan]*4)
    
    # 生成MCMC速度序列
    speeds = mcmc_speed_generator(target_mean, n, max_accel=2.5)
    
    # 
    split_accel = int(n * 0.4)
    split_decel = n - int(n * 0.4)
    
    # 
    accel_segment = speeds[:split_accel]
    decel_segment = speeds[split_decel:]
    
    return pd.Series([
        accel_segment.mean(), accel_segment.var(),
        decel_segment.mean(), decel_segment.var()
    ])

# 使用示例
# processed_arr_mcmc = process_speed_data(arr, generator=generate_curve_mcmc)
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
        generated=arr.apply(generate_curve_mcmc, axis=1)

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

        # 定义梯度提升树的参数网格
        param_grid = {
            'n_estimators': [5, 10, 20],
            'learning_rate': [0.01, 0.1, 0.3],
            # 'n_estimators': [10],
            # 'learning_rate': [0.1],
            # 其他参数也可以在此添加
        }

        # 初始化梯度提升树回归器
        xgb_regressor = xgb.XGBRegressor(random_state=42)

        # 在内部循环中进行网格搜索
        grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=2)
        grid_search.fit(trainX, trainY)

        # # 使用最佳参数的模型进行预测
        best_xgb_regressor = grid_search.best_estimator_
        EC_Pred = best_xgb_regressor.predict(testX)

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
# 保存为 pickle 文件
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/XGB_SUM_M.pkl', 'wb') as f:
    pickle.dump(SUM_M, f)
#%%

#%%
#%%
data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/XGB_data.pkl', 'wb') as f:
    pickle.dump(data, f)

#%% for scatter
# indices_to_remove = [0, 15, 16, 17, 18]

# # Iterate over the indices in reverse order to avoid index shifting issues while deleting
# for index in sorted(indices_to_remove, reverse=True):
#     del test_all_simu[index]
#     del test_all_real[index]

# EC_Pred = np.concatenate(test_all_simu, axis=0)
# EC_True = np.concatenate(test_all_real, axis=0)

# data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# # 保存为 pickle 文件
# with open('11-pretrained_incremental_learning/data_plot/XGB_scatter.pkl', 'wb') as f:
#     pickle.dump(data, f)
# #%%
# """Charge Car"""
# with open("11-pretrained_incremental_learning/data/ChargeCar.pkl", "rb") as f:
#     ChargeCar = pickle.load(f)
# x = ChargeCar.values[:, 1:]
# y = ChargeCar.values[:, 0]

# X_train, X_test, y_train, y_test = \
#     train_test_split(x, y, test_size=0.9, shuffle=False)

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.3],
#     # 其他参数也可以在此添加
# }

# # 初始化梯度提升树回归器
# xgb_regressor = xgb.XGBRegressor(random_state=42)

# # 在内部循环中进行网格搜索
# grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid,
#                            scoring='neg_mean_squared_error', cv=5)
# grid_search.fit(X_train, y_train)

# # 使用最佳参数的模型进行预测
# best_xgb_regressor = grid_search.best_estimator_
# EC_Pred = best_xgb_regressor.predict(X_test)

# EC_True = y_test

# Metric1 = np.array(evaluation(EC_True, EC_Pred))
# print("acc:", Metric1)

# data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# # 保存为 pickle 文件
# with open('11-pretrained_incremental_learning/data_plot/XGB_Chargecar.pkl', 'wb') as f:
#     pickle.dump(data, f)
# #%%
# """BHD"""

# with open("11-pretrained_incremental_learning/data/BHD.pkl", "rb") as f:
#     BHD = pickle.load(f)


# rename_dict = {
#     'Date': '出行时间',
#     'Distance [km]': '行程距离',
#     'Duration [min]': '行程时间',
#     'Battery State of Charge (Start)': '当前SOC',
#     'Ambient Temperature (Start) [°C]': 'T'
# }
# BHD.rename(columns=rename_dict, inplace=True)
# def labeling2(Array):
#     """label encoding"""
#     # 对出行季节进行Label Encoding
#     season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
#     Array['出行季节'] = Array['出行季节'].map(season_mapping)

#     # 对出行日期进行Label Encoding
#     day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
#                    'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
#     Array['出行日期'] = Array['出行日期'].map(day_mapping)

#     # 对出行时段进行Label Encoding
#     period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
#     Array['出行时段'] = Array['出行时段'].map(period_mapping)

#     # 对车辆类型进行Label Encoding
#     vehicle_mapping = {'Sedan': 0, 'SUV': 0.333, 'Sedan PHEV': 0.667, 'SUV PHEV': 1}
#     Array['车辆类型'] = Array['车辆类型'].map(vehicle_mapping)

#     Array['整备质量'] = Array['整备质量']/1880
#     Array['电池能量'] = Array['电池能量'] /61.1


#     columns_to_drop = ["出行时间", "地点"]
#     Array = Array.drop(columns=columns_to_drop).astype(float)
#     Array = Array.fillna(0)

#     return Array

# BHD=labeling2(BHD)

# BHD['当前SOC']=BHD['当前SOC']*100

# x = BHD.values[:, 1:]
# y = BHD.values[:, 0]


# X_train, X_test, y_train, y_test = \
#     train_test_split(x, y, test_size=0.9, shuffle=False)

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.3],
#     # 其他参数也可以在此添加
# }

# # 初始化梯度提升树回归器
# xgb_regressor = xgb.XGBRegressor(random_state=42)

# # 在内部循环中进行网格搜索
# grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid,
#                            scoring='neg_mean_squared_error', cv=5)
# grid_search.fit(X_train, y_train)

# # 使用最佳参数的模型进行预测
# best_xgb_regressor = grid_search.best_estimator_
# EC_Pred = best_xgb_regressor.predict(X_test)

# EC_True = y_test

# Metric1 = np.array(evaluation(EC_True, EC_Pred))
# print("acc:", Metric1)

# data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# # 保存为 pickle 文件
# with open('11-pretrained_incremental_learning/data_plot/XGB_BHD.pkl', 'wb') as f:
#     pickle.dump(data, f)
# #%%
# """SpritMonitor"""

# with open("11-pretrained_incremental_learning/data/SpritMonitor.pkl", "rb") as f:
#     SpritMonitor = pickle.load(f)


# rename_dict = {
#     'quantity(kWh)': '行程能耗',
#     'trip_distance(km)': '行程距离',
#     'odometer': '当前累积行驶里程',
#     'fuel_date': '出行时间',
#     'avg_speed(km/h)': '行程平均速度',
# }


# def labeling(df):
#     """label encoding"""
#     # 对出行季节进行 Label Encoding
#     season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
#     df['出行季节'] = df['出行季节'].map(season_mapping)

#     # 对出行日期进行 Label Encoding
#     day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
#                    'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
#     df['出行日期'] = df['出行日期'].map(day_mapping)

#     # 对出行时段进行 Label Encoding
#     period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
#     df['出行时段'] = df['出行时段'].map(period_mapping)

#     # 对车辆类型进行 Label Encoding
#     vehicle_mapping = {'Sedan': 0, 'SUV': 0.333, 'Sedan PHEV': 0.667, 'SUV PHEV': 1}
#     df['车辆类型'] = df['车辆类型'].map(vehicle_mapping)

#     # 对驾驶风格进行 Label Encoding
#     style_mapping = {'Normal': 0, 'Moderate': 0.5, 'Fast': 1}
#     df['driving_style'] = df['driving_style'].map(style_mapping)

#     # 对轮胎类型进行 Label Encoding
#     tire_mapping = {'Winter tires': 0, 'Summer tires': 1}
#     df['tire_type'] = df['tire_type'].map(tire_mapping)

#     df['整备质量'] = df['整备质量'] / 1880
#     df['电池能量'] = df['电池能量'] / 61.1

#     # 删除不需要的列
#     columns_to_drop = ["出行时间", "地点"]
#     df.drop(columns=columns_to_drop, inplace=True)

#     # 将数据类型转换为 float，并填充缺失值为 0
#     df = df.astype(float).fillna(0)

#     return df

# # 对每个 DataFrame 执行重命名和标签编码操作
# SpritMonitor_labeled=[]
# for df in SpritMonitor:
#     df.rename(columns=rename_dict, inplace=True)  # 执行重命名操作
#     SpritMonitor_labeled.append(labeling(df))  # 执行标签编码操作
# A=[]
# B=[]
# for structured in SpritMonitor_labeled:

#     x = structured.values[:, 1:]
#     y = structured.values[:, 0]

#     X_train, X_test, y_train, y_test = \
#         train_test_split(x, y, test_size=0.9, shuffle=False)

#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 0.3],
#         # 其他参数也可以在此添加
#     }

#     # 初始化梯度提升树回归器
#     xgb_regressor = xgb.XGBRegressor(random_state=42)

#     # 在内部循环中进行网格搜索
#     grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid,
#                                scoring='neg_mean_squared_error', cv=5)
#     grid_search.fit(X_train, y_train)

#     # 使用最佳参数的模型进行预测
#     best_xgb_regressor = grid_search.best_estimator_
#     EC_Pred = best_xgb_regressor.predict(X_test)

#     EC_True = y_test

#     Metric1 = np.array(evaluation(EC_True, EC_Pred))
#     print("acc:", Metric1)
#     A.append(EC_True)
#     B.append(EC_Pred)

# EC_True = np.hstack(A).reshape(-1, 1)
# EC_Pred = np.hstack(B).reshape(-1, 1)
# Metric = np.array(evaluation(EC_True, EC_Pred))
# print("acc:", Metric)
# #%%
# data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# # 保存为 pickle 文件
# with open('11-pretrained_incremental_learning/data_plot/XGB_SpritMonitor.pkl', 'wb') as f:
#     pickle.dump(data, f)
# #%%
# """VED"""

# with open("11-pretrained_incremental_learning/data/VED.pkl", "rb") as f:
#     VED = pickle.load(f)
# VED=pd.concat(VED, ignore_index=True)
# rename_dict = {
#     '湿度': 'U',
#     '可见度': 'VV',
#     '风速': 'Ff',
#     '温度': 'T',
#     '降雨': 'RRR',
# }

# VED.rename(columns=rename_dict, inplace=True)  # 执行重命名操作

# VED["VV"] = VED["VV"].replace("2.50V", "2.5")

# def labeling(df):
#     """label encoding"""
#     # 对出行季节进行 Label Encoding
#     season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
#     df['出行季节'] = df['出行季节'].map(season_mapping)

#     # 对出行日期进行 Label Encoding
#     day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
#                    'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
#     df['出行日期'] = df['出行日期'].map(day_mapping)

#     # 对出行时段进行 Label Encoding
#     period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
#     df['出行时段'] = df['出行时段'].map(period_mapping)

#     # 对车辆类型进行 Label Encoding
#     vehicle_mapping = {'Sedan': 0, 'SUV': 0.333, 'Sedan PHEV': 0.667, 'SUV PHEV': 1}
#     df['车辆类型'] = df['车辆类型'].map(vehicle_mapping)

#     df['整备质量'] = df['整备质量'] / 1880
#     df['电池能量'] = df['电池能量'] / 61.1

#     # 删除不需要的列
#     columns_to_drop = ["出行时间", "地点"]
#     df.drop(columns=columns_to_drop, inplace=True)

#     # 将数据类型转换为 float，并填充缺失值为 0
#     df = df.astype(float).fillna(0)

#     return df


# VED=labeling(VED) # 执行标签编码操作

# x = VED.values[:, 1:]
# y = VED.values[:, 0]

# X_train, X_test, y_train, y_test = \
#     train_test_split(x, y, test_size=0.9, shuffle=False)

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.3],
#     # 其他参数也可以在此添加
# }

# # 初始化梯度提升树回归器
# xgb_regressor = xgb.XGBRegressor(random_state=42)

# # 在内部循环中进行网格搜索
# grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid,
#                            scoring='neg_mean_squared_error', cv=2)
# grid_search.fit(X_train, y_train)

# # 使用最佳参数的模型进行预测
# best_xgb_regressor = grid_search.best_estimator_
# EC_Pred = best_xgb_regressor.predict(X_test)

# EC_True = y_test

# Metric1 = np.array(evaluation(EC_True, EC_Pred))
# print("acc:", Metric1)

# data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# # 保存为 pickle 文件
# with open('11-pretrained_incremental_learning/data_plot/XGB_VED.pkl', 'wb') as f:
#     pickle.dump(data, f)

# #%%
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.plot(EC_True, 'ro-', label='True Values')  # 真实值用红点表示
# plt.plot(EC_Pred, 'k*-', label='Predicted Values')  # 预测值用黑线表示
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title('True Values vs Predicted Values')
# plt.legend()
# plt.grid(True)
# plt.show()
