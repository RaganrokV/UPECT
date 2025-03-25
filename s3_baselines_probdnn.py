# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
from My_utils.evaluation_scheme import evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor

#%%
"""
neural networks
Data-Driven Probabilistic Energy Consumption Estimation 
for Battery Electric Vehicles with Model Uncertainty
"""
#%%
import torch
import torch.nn as nn
import torch.optim as optim
# from skorch import NeuralNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
#%%
"""local"""
with open('/home/ps/haichao/11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)


with open('/home/ps/haichao/11-pretrained_incremental_learning/normalization_params.pkl', 'rb') as f:
    min_values, max_values = pickle.load(f)


def labeling(Array):
    """label encoding"""
    # Label encoding for different categorical variables
    season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
    Array['出行季节'] = Array['出行季节'].map(season_mapping)

    day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
                   'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
    Array['出行日期'] = Array['出行日期'].map(day_mapping)

    period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
    Array['出行时段'] = Array['出行时段'].map(period_mapping)

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
# 定义概率MLP模型
class ProbabilisticMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes):
        super().__init__()
        self.shared_layers = nn.ModuleList()
        in_size = input_dim
        for out_size in hidden_layer_sizes:
            self.shared_layers.append(nn.Linear(in_size, out_size))
            self.shared_layers.append(nn.ReLU())
            in_size = out_size
        
        # 均值输出层
        self.mean_layer = nn.Linear(in_size, 1)
        # 方差输出层（使用Softplus确保正值）
        self.var_layer = nn.Sequential(
            nn.Linear(in_size, 1),
            nn.Softplus()
        )

    def forward(self, X):
        for layer in self.shared_layers:
            X = layer(X)
        return self.mean_layer(X), self.var_layer(X)

# 自定义负对数似然损失
def nll_loss(y_pred, y_true):
    mu, var = y_pred
    return 0.5 * torch.mean(torch.log(var) + (y_true - mu)**2 / var)


# 修改后的训练循环
SUM_M = []
P_all = []
for p in [32]:
    M = []
    test_all_simu = []
    test_all_real = []
    for arr in Test_vehicle:
        # 数据预处理保持不变...
        Array = labeling(arr)
        np.random.seed(42)
        Normalized_array = (Array - min_values) / (max_values - min_values)
        Normalized_array = Normalized_array.reset_index(drop=True)
        
        # 划分数据集
        train_size = p
        test_size = int(len(Normalized_array) * 0.3)
        trainX = Normalized_array.iloc[:train_size, 1:].values.astype(np.float32)
        trainY = Normalized_array.iloc[:train_size, 0].values.astype(np.float32)
        testX = Normalized_array.iloc[-test_size:, 1:].values.astype(np.float32)
        testY = Normalized_array.iloc[-test_size:, 0].values.astype(np.float32)

        # 处理单样本情况
        if len(trainX) == 1:
            trainX = np.vstack([trainX, trainX])
            trainY = np.append(trainY, trainY[0])

        # 转换为PyTorch张量
        X_tensor = torch.from_numpy(trainX)
        y_tensor = torch.from_numpy(trainY).view(-1, 1)
        
        # 初始化模型
        input_dim = trainX.shape[1]
        model = ProbabilisticMLP(input_dim=input_dim, hidden_layer_sizes=(32,32))
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        
        # 训练循环
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            mu_pred, var_pred = model(X_tensor)
            loss = nll_loss((mu_pred, var_pred), y_tensor)
            loss.backward()
            optimizer.step()
        
        # 预测
        model.eval()
        with torch.no_grad():
            test_tensor = torch.from_numpy(testX)
            mu_pred, _ = model(test_tensor)  # 只取均值用于评估
            
        # 逆归一化
        EC_True = testY * (max_values[0] - min_values[0]) + min_values[0]
        EC_Pred = mu_pred.numpy() * (max_values[0] - min_values[0]) + min_values[0]
        
        # 评估
        Metric1 = np.array(evaluation(EC_True, EC_Pred))
        print( Metric1)
        M.append(Metric1)
        test_all_simu.append(EC_Pred.reshape(-1, 1))
        test_all_real.append(EC_True.reshape(-1, 1))

    # 后续评估保持不变...
    M2_test_all = np.array(evaluation(np.vstack(test_all_real), np.vstack(test_all_simu)))
    P_all.append(M2_test_all)
    METRIC = np.vstack(M)
    SUM_M.append(METRIC)

pd.DataFrame(SUM_M[-1])
#%%
data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/NLPnn_data.pkl', 'wb') as f:
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
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/MLPnn_scatter.pkl', 'wb') as f:
    pickle.dump(data, f)
