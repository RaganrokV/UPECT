# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from My_utils.evaluation_scheme import evaluation
import math
import torch.nn.parallel

import time
#%% load
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

        self.Feat_embedding = nn.Linear(1, 512, bias=False) # equal to nn.embedding

        self.pos = PositionalEncoding(512,max_len=100)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512,
                                                        nhead=8,
                                                        dim_feedforward=2048,
                                                        batch_first=True,
                                                        dropout=0.1,
                                                        activation="gelu")

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=12)


        self.feat_map = nn.Linear(20*512, 1, bias=True)

        self.out_fc = nn.Linear(512*20, 1, bias=True)
        self.activation = nn.ReLU()  # 添加激活函数
        self.dropout = nn.Dropout(0.1)  # 添加Dropout层
        self.bn = nn.BatchNorm1d(20)

        """only for transfer learning, please do not compile this code """

        """if you wanna a full fine tuning,just run as pretraining.
            for selective fine tuning, run following code-1 """
        """1.refreeze the last layer """
        # for param in self.parameters():
        #     param.requires_grad = False
        #
        # # 解冻最后的全连接层
        # self.out_fc.weight.requires_grad = True
        # self.out_fc.bias.requires_grad = True

        """2.refreeze the first and last layer """
        # # 解冻最后的全连接层和Feat_embedding
        # for param in self.parameters():
        #     param.requires_grad = True
        #
        # # 冻结transformer_encoder的参数
        # for param in self.transformer_encoder.parameters():
        #     param.requires_grad = False

        """  """

    def forward(self, src):


        B,F=src.size()
        # embedding
        embedding_src=self.Feat_embedding(src.unsqueeze(2)) #(128,20,1)--(128,20,512)

        embed_encoder_input = self.pos(embedding_src) #essential add a shift

        # transform
        out = self.transformer_encoder(embed_encoder_input) #(128,20,512)

        x = self.bn(out)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_fc(x.reshape(B,-1))

        return x,embedding_src


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
#%%
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


#%%
"""
SFT(Supervised Fine-Tuning),Full-Param Fine-tuning

or 

transfer learning,Freeze Fine-tuning (only update the last layer)
"""


with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

with open('11-pretrained_incremental_learning/normalization_params.pkl', 'rb') as f:
    min_values, max_values = pickle.load(f)

test_M=[]
test_all_simu = []
test_all_real = []
train_time=[]
inference_time=[]
for arr in Test_vehicle:

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

    Array=labeling(arr)
    Normalized_array = (Array - min_values) / (max_values - min_values)

    train_size=int(len(Normalized_array)*0.7)

    trainX = torch.Tensor(Normalized_array.iloc[:train_size, 1:].values).float()
    trainY = torch.Tensor(Normalized_array.iloc[:train_size, 0].values).float()

    testX = torch.Tensor(Normalized_array.iloc[train_size:, 1:].values).float()
    testY = torch.Tensor(Normalized_array.iloc[train_size:, 0].values).float()

    """lr =1e-5 for Full-Param Fine-tuning
        lr=1e-4 for transfer learning       """
    optimizer = torch.optim.AdamW(UPECT.parameters(), lr=1e-5,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=2, factor=0.99)
    criterion = nn.MSELoss()

    # 记录训练开始时间
    train_start_time = time.time()

    num_epochs = 5
    for epoch in range(num_epochs):
        UPECT.to(device).train()
        total_loss = 0.0
        trainX, trainY = trainX.to(device), trainY.to(device)

        optimizer.zero_grad()
        pre_y, _ = UPECT(trainX)
        loss = criterion(pre_y, trainY.unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * trainX.size(0)

        epoch_loss = total_loss / len(trainX)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 记录训练结束时间
    train_end_time = time.time()

    # 计算训练时间
    train_time.append((train_end_time - train_start_time)/trainX.size(0))


    # 记录推断开始时间
    inference_start_time = time.time()

    UPECT.to(device).eval()
    with torch.no_grad():
        pred, _ = UPECT(testX.to(device))
        predictions = pred.data.cpu().numpy()
        targets = testY.cpu().numpy()

    # 记录推断结束时间
    inference_end_time = time.time()

    # 计算推断时间
    inference_time.append((inference_end_time - inference_start_time)/testX.size(0))


    EC_True = ((targets) * (max_values[0] - min_values[0]) / 2) + min_values[0]
    EC_Pred = ((np.abs(predictions)) * (max_values[0] - min_values[0]) / 2) + min_values[0]
    Metric1 = np.array(evaluation(EC_True, EC_Pred))
    print("acc:", Metric1)
    test_M.append(Metric1)
    test_all_simu.append(EC_Pred.reshape(-1, 1))
    test_all_real.append(EC_True.reshape(-1, 1))

    del optimizer,scheduler,UPECT


M2_test_ind=np.vstack(test_M)
M2_test_all=np.array(evaluation(np.vstack(test_all_real),
                                 np.vstack(test_all_simu)))

train_time_mean = sum(train_time) / len(train_time)
print("train Time:", train_time_mean)

inference_time_mean = sum(inference_time) / len(inference_time)
print("Inference Time:", inference_time_mean)




