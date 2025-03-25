# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
import warnings
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from My_utils.evaluation_scheme import evaluation

warnings.filterwarnings("ignore")
#%%
# Load data
with open('/home/ps/haichao/11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

with open('/home/ps/haichao/11-pretrained_incremental_learning/normalization_params.pkl', 'rb') as f:
    min_values, max_values = pickle.load(f)


def labeling(Array):
    """Label encoding"""
    season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
    Array['出行季节'] = Array['出行季节'].map(season_mapping)

    day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
                   'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
    Array['出行日期'] = Array['出行日期'].map(day_mapping)

    period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
    Array['出行时段'] = Array['出行时段'].map(period_mapping)

    vehicle_mapping = {'Sedan': 0, 'SUV': 0.333, 'Sedan PHEV': 0.667, 'SUV PHEV': 1}
    Array['车辆类型'] = Array['车辆类型'].map(vehicle_mapping)

    Array['整备质量'] = Array['整备质量'] / 1880
    Array['电池能量'] = Array['电池能量'] / 61.1
    Array['当前累积行驶里程'] = Array['当前累积行驶里程'] / 500000

    Array.loc[Array['VV'].apply(lambda x: isinstance(x, str)), 'VV'] = 0.1

    columns_to_drop = ["出行时间", "地点", "VIN"]
    Array = Array.drop(columns=columns_to_drop).astype(float)
    Array = Array.fillna(0)

    return Array

# Define BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # Multiply by 2 for bidirectional

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the last time step output for prediction
        out = self.fc(lstm_out[:, -1, :])
        return out

SUM_M = []
P_all = []

for p in [32]:
    M = []
    test_all_simu = []
    test_all_real = []
    for arr in Test_vehicle:
        # arr = Test_vehicle[5]
        Array = labeling(arr)
        np.random.seed(42)

        Normalized_array = (Array - min_values) / (max_values - min_values)

        Normalized_array = Normalized_array.reset_index(drop=True)
        x = Array.values[:, 1:]
        y = Array.values[:, 0]

        train_size = p
        test_size = int(len(x) * 0.3)

        trainX = Normalized_array .iloc[:train_size, 1:].values
        trainY = Normalized_array .iloc[:train_size, 0].values
        testX = Normalized_array .iloc[-test_size:, 1:].values
        testY = Normalized_array .iloc[-test_size:, 0].values

        # Convert data to PyTorch tensors and reshape for LSTM (samples, time_steps, features)
        trainX = torch.tensor(trainX, dtype=torch.float32).unsqueeze(1)
        trainY = torch.tensor(trainY, dtype=torch.float32)
        testX = torch.tensor(testX, dtype=torch.float32).unsqueeze(1)
        testY = torch.tensor(testY, dtype=torch.float32)

        # Create data loaders
        train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=8, shuffle=True)

        # Define model, loss, and optimizer
        input_size = trainX.shape[2]
        model = BiLSTM(input_size=input_size, hidden_size=32, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        model.train()
        for epoch in range(50):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            EC_Pred = model(testX).squeeze().numpy()
            EC_True = testY.numpy()

        EC_True = ((EC_True) * (max_values[0] - min_values[0])) + min_values[0]
        EC_Pred = ((np.abs(EC_Pred)) * (max_values[0] - min_values[0])) + min_values[0]
        # Calculate accuracy metrics
        Metric1 = np.array(evaluation(EC_True, EC_Pred))
        print(Metric1)
        M.append(Metric1)

        test_all_simu.append(EC_Pred.reshape(-1, 1))
        test_all_real.append(EC_True.reshape(-1, 1))
        # break

    M2_test_all = np.array(evaluation(np.vstack(test_all_real), np.vstack(test_all_simu)))
    P_all.append(M2_test_all)

    METRIC = np.vstack(M)
    SUM_M.append(METRIC)

pd.DataFrame(SUM_M[-1])
#%%
data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/BiLSTM_data.pkl', 'wb') as f:
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
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/BiLSTM_scatter.pkl', 'wb') as f:
    pickle.dump(data, f)