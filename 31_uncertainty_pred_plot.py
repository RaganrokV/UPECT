# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from My_utils.evaluation_scheme import evaluation
import math
import torch.nn.parallel

import time


# %% load
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe.requires_grad = False
        pe = pe.unsqueeze(0)  # 在批次维度上增加维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]  # 对位置编码进行广播并添加到输入张量上


class Transformer_encoder(nn.Module):
    def __init__(self):
        super(Transformer_encoder, self).__init__()

        self.Feat_embedding = nn.Linear(1, 512, bias=False)  # equal to nn.embedding

        self.pos = PositionalEncoding(512, max_len=100)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512,
                                                        nhead=8,
                                                        dim_feedforward=2048,
                                                        batch_first=True,
                                                        dropout=0.1,
                                                        activation="gelu")

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=12)

        self.feat_map = nn.Linear(20 * 512, 1, bias=True)

        self.out_fc = nn.Linear(512 * 20, 1, bias=True)
        self.activation = nn.ReLU()  # 添加激活函数
        self.dropout = nn.Dropout(0.1)  # 添加Dropout层
        self.bn = nn.BatchNorm1d(20)


    def forward(self, src, use_dropout=False):
        B, _ = src.size()
        # embedding
        embedding_src = self.Feat_embedding(src.unsqueeze(2))  # (128,20,1)--(128,20,512)

        embed_encoder_input = self.pos(embedding_src)  # essential add a shift

        # transform
        out = self.transformer_encoder(embed_encoder_input)  # (128,20,512)

        x = self.bn(out)
        x = self.activation(x)
        # Apply dropout conditionally
        if use_dropout:
            # Generate a random dropout probability between 0.1 and 0.9
            p = torch.rand(1).item() * 0.8 + 0.1
            x = F.dropout(x, p=p, training=True)  # Dropout during inference with random p # Dropout during inference if requested
        else:
            x = self.dropout(x)  # Standard dropout during training

        # # x = self.dropout(x)
        # if use_dropout:
        #     x = F.dropout(x, p=0.5, training=True)  # Dropout during inference if requested

        x = self.out_fc(x.reshape(B, -1))  # Output layer

        return x


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


# %%
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

# %%
"""
SFT(Supervised Fine-Tuning),Full-Param Fine-tuning

or 

transfer learning,Freeze Fine-tuning (only update the last layer)
"""

for p in [0.7]:
    with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
        Test_vehicle = pickle.load(file)

    with open('11-pretrained_incremental_learning/normalization_params.pkl', 'rb') as f:
        min_values, max_values = pickle.load(f)

    test_M = []
    test_all_simu = []
    test_all_real = []
    train_time = []
    inference_time = []
    for arr in Test_vehicle:

        arr=Test_vehicle[10]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        UPECT = Transformer_encoder()
        ckpt = torch.load('11-pretrained_incremental_learning/model/UPECT_40M.pt')

        # 提取 DataParallel 包装的模型中的实际模型参数
        state_dict = ckpt["model_state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 去除 'module.' 的前缀
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        UPECT.load_state_dict(new_state_dict)

        Array = labeling(arr)
        Normalized_array = (Array - min_values) / (max_values - min_values)

        Normalized_array = Normalized_array.sample(frac=1).reset_index(drop=True)
        train_size = int(len(Normalized_array) * p)
        # train_size = 1
        test_size = int(len(Normalized_array) * 0.3)

        trainX = torch.Tensor(Normalized_array.iloc[:train_size, 1:].values).float()
        trainY = torch.Tensor(Normalized_array.iloc[:train_size, 0].values).float()


        testX = torch.Tensor(Normalized_array.iloc[-test_size:, 1:].values).float()
        testY = torch.Tensor(Normalized_array.iloc[-test_size:, 0].values).float()

        """lr =1e-5 for Full-Param Fine-tuning
            lr=1e-4 for transfer learning       """
        optimizer = torch.optim.AdamW(UPECT.parameters(), lr=1e-5,
                                      betas=(0.9, 0.999), weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        criterion = nn.MSELoss()

        # 记录训练开始时间
        train_start_time = time.time()

        num_epochs = 5
        for epoch in range(num_epochs):
            UPECT.to(device).train()
            total_loss = 0.0
            trainX, trainY = trainX.to(device), trainY.to(device)

            optimizer.zero_grad()
            pre_y = UPECT(trainX)
            loss = criterion(pre_y, trainY.unsqueeze(1))
            loss.backward()
            optimizer.step()


            total_loss += loss.item() * trainX.size(0)

            epoch_loss = total_loss / len(trainX)
            current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Learning Rate: {current_lr:.6f}")
            # scheduler.step()

        # 记录训练结束时间
        train_end_time = time.time()

        # 计算训练时间
        train_time.append((train_end_time - train_start_time) / trainX.size(0))

        # 记录推断开始时间
        inference_start_time = time.time()

        UPECT.to(device).eval()
        mul_predictions = []
        for _ in range(50):
            with torch.no_grad():
                pred =UPECT(testX.to(device), use_dropout=True)
                predictions = pred.data.cpu().numpy()
                targets = testY.cpu().numpy()
                mul_predictions.append(predictions)

            # pred = UPECT(testX.to(device))
            # predictions = pred.data.cpu().numpy()
            # targets = testY.cpu().numpy()



        # 记录推断结束时间
        inference_end_time = time.time()

        # 计算推断时间
        inference_time.append((inference_end_time - inference_start_time) / test_size)


        del optimizer,  UPECT

        break



#%%
predictions = np.array(mul_predictions)
mean_array = np.mean(predictions, axis=0)
uncertainty1 = np.std(predictions, axis=0)
uncertainty2 = np.std(predictions[-10:], axis=0)
lower_bound = mean_array - 2.576 * np.sqrt(uncertainty1 ** 2 + uncertainty2 ** 2)
upper_bound = mean_array + 2.576 * np.sqrt(uncertainty1 ** 2 + uncertainty2 ** 2)

EC_True = ((targets) * (max_values[0] - min_values[0]) / 2) + min_values[0]
EC_Pred = ((np.abs(mean_array)) * (max_values[0] - min_values[0]) / 2) + min_values[0]
EC_lower_bound=((lower_bound) * (max_values[0] - min_values[0]) / 2) + min_values[0]
EC_upper_bound=((upper_bound) * (max_values[0] - min_values[0]) / 2) + min_values[0]


#%%
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle
# 创建 DataFrame
df = pd.DataFrame({
    'Index': np.arange(len(EC_True)),
    'Real': EC_True,
    'Mean': EC_Pred.flatten(),
    'EC_lower_bound': EC_lower_bound.flatten(),
    'EC_upper_bound': EC_upper_bound.flatten()
})

# 绘制主图
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(15, 6))

# 绘制真实值
sns.scatterplot(data=df, x='Index', y='Real', label='Real', color='b', ax=ax)

# 绘制预测均值
sns.lineplot(data=df, x='Index', y='Mean', label='Mean predicted', color='r', ax=ax)

# 绘制置信区间
ax.fill_between(df['Index'], df['EC_lower_bound'], df['EC_upper_bound'],
                 color='blue', alpha=0.2, label='99% confidence interval')

# 设置坐标轴标签
ax.set_xlabel('Trips over time', fontsize=25)
ax.set_ylabel('Energy consumption (kWh)', fontsize=25)


# 设置放大区域
x1, x2, y1, y2 = 90, 110, -1, 18

# 创建插图区域
axins = inset_axes(ax, width="55%", height="40%", loc='upper right')
# 绘制相同的数据到插图区域
axins.plot(df['Index'], df['Real'], linestyle='None', marker='o',
           markersize=8, label='Real values', color='b')
axins.plot(df['Index'], df['Mean'], label='Mean predictions', color='r')
axins.fill_between(df['Index'], df['EC_lower_bound'], df['EC_upper_bound'],
                 color='r', alpha=0.2, label='99% confidence interval')

# 设置插图区域的坐标轴范围
axins.set_xlim(x1, x2)
axins.set_ylim(-2, y2)

# 去除插图区域的坐标轴刻度
axins.set_xticks([])
axins.set_yticks([])

# 在主图上标记插图的位置
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k', lw=3, linestyle='--')

# 绘制红色矩形框突出显示放大区域
rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3,
                 edgecolor='k', facecolor='none', linestyle='-')
ax.add_patch(rect)

# 设置 x 和 y 轴刻度标签大小
plt.legend(fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

# 调整布局
plt.tight_layout()
plt.savefig('11-pretrained_incremental_learning/Fig/uncertainty.svg', dpi=600)
plt.savefig('11-pretrained_incremental_learning/Fig/uncertainty.png', dpi=600)
plt.show()

