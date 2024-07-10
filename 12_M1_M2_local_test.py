# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.utils.data as Data
from My_utils.evaluation_scheme import evaluation
import math
import torch.nn.parallel

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


class Transformer_src2tgt(nn.Module):
    def __init__(self):
        super(Transformer_src2tgt, self).__init__()

        self.Feat_embedding = nn.Linear(1, 512, bias=False) # equal to nn.embedding
        self.src2tgt = nn.Linear(20, 1, bias=True)
        self.pos = PositionalEncoding(512,max_len=100)
        self.transform = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.5,
            activation="gelu",
        )
        self.out_fc = nn.Linear(512, 1, bias=True)
        self.activation = nn.ReLU()  # 添加激活函数
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.bn = nn.BatchNorm1d(512)


    def forward(self, src):

        # embedding
        embedding_src=self.Feat_embedding(src.unsqueeze(2))

        tgt=self.src2tgt(src)

        embedding_tgt = self.Feat_embedding(tgt.unsqueeze(2))

        embed_encoder_input = self.pos(embedding_src) #essential add a shift
        embed_decoder_input = self.pos(embedding_tgt)  # essential add a shift

        # transform
        out = self.transform(embed_encoder_input, embed_decoder_input) #(128,1,512)

        x = self.bn(out.squeeze())
        x = self.activation(x)
        x = self.out_fc(x)


        return x,embedding_src


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
"""Model 2"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

UPECM2 = Transformer_encoder()
ckpt = torch.load('11-pretrained_incremental_learning/model/UPECT_40M.pt')
# ckpt = torch.load('11-pretrained_incremental_learning/model/model_checkpoint_M2_EP3000.pt')
# ckpt = torch.load('11-pretrained_incremental_learning/model/model_checkpoint.pt')
# 提取 DataParallel 包装的模型中的实际模型参数
state_dict = ckpt["model_state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_key = key[7:]  # 去除 'module.' 的前缀
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

UPECM2.load_state_dict(new_state_dict)
#%%
with open('11-pretrained_incremental_learning/data/Train_vehicle.pkl', 'rb') as file:
    Train_vehicle = pickle.load(file)

with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

with open('11-pretrained_incremental_learning/normalization_params.pkl', 'rb') as f:
    min_values, max_values = pickle.load(f)

HDP=[]
test_M=[]
test_all_simu = []
test_all_real = []
for arr in Test_vehicle:

    Array=labeling(arr)
    Normalized_array = (Array - min_values) / (max_values - min_values)

    testX = torch.Tensor(Normalized_array.iloc[:, 1:].values).float()
    testY = torch.Tensor(Normalized_array.iloc[:, 0].values).float()

    UPECM2.to(device).eval()
    with torch.no_grad():
        pred, HP = UPECM2(testX.to(device))
        predictions = pred.data.cpu().numpy()
        targets=testY.cpu().numpy()

    EC_True = ((targets) * (max_values[0] - min_values[0]) / 2) + min_values[0]
    EC_Pred = ((np.abs(predictions)) * (max_values[0] - min_values[0]) / 2) + min_values[0]
    Metric1 = np.array(evaluation(EC_True, EC_Pred))
    print("acc:", Metric1)
    test_M.append(Metric1)
    test_all_simu.append(EC_Pred.reshape(-1,1))
    test_all_real.append(EC_True.reshape(-1,1))
    HDP.append(HP)

M2_test_ind=np.vstack(test_M)
M2_test_all=np.array(evaluation(np.vstack(test_all_real),
                                 np.vstack(test_all_simu)))


# train_all_simu = []
# train_all_real = []
# train_M=[]
# for arr in Train_vehicle:
#
#     if len(arr) <= 1:
#         continue
#
#     Array=labeling(arr)
#     Normalized_array = (Array - min_values) / (max_values - min_values)
#
#     testX = torch.Tensor(Normalized_array.iloc[:, 1:].values).float()
#     testY = torch.Tensor(Normalized_array.iloc[:, 0].values).float()
#
#     GPECM2.to(device).eval()
#     with torch.no_grad():
#         pred, _ = GPECM2(testX.to(device))
#         predictions = pred.data.cpu().numpy()
#         targets=testY.cpu().numpy()
#
#     EC_True = ((targets) * (max_values[0] - min_values[0]) / 2) + min_values[0]
#     EC_Pred = ((np.abs(predictions)) * (max_values[0] - min_values[0]) / 2) + min_values[0]
#     Metric1 = np.array(evaluation(EC_True, EC_Pred))
#     print("acc:", Metric1)
#     train_M.append(Metric1)
#     train_all_simu.append(EC_Pred.reshape(-1,1))
#     train_all_real.append(EC_True.reshape(-1,1))
#
#
# M2_train_ind=np.mean(np.vstack(train_M),axis=0)
# M2_train_all=np.array(evaluation(np.vstack(train_all_real),
#                                  np.vstack(train_all_simu)))
#%%
"""save data for TSNE"""
P=torch.vstack(HDP).cpu()
Embeddings = P.reshape(13221, 20 * 512)
with open('11-pretrained_incremental_learning/data/Embeddings.pkl', 'wb') as f:
    pickle.dump(Embeddings, f)

# """Model src2tgt"""

# # Initialize your Transformer model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
#
# GPECM1 = Transformer_src2tgt()
# ckpt = torch.load('11-pretrained_incremental_learning/model/model_checkpoint_BN512.pt')
#
# # 提取 DataParallel 包装的模型中的实际模型参数
# state_dict = ckpt["model_state_dict"]
# new_state_dict = {}
# for key, value in state_dict.items():
#     if key.startswith('module.'):
#         new_key = key[7:]  # 去除 'module.' 的前缀
#         new_state_dict[new_key] = value
#     else:
#         new_state_dict[key] = value
#
# GPECM1.load_state_dict(new_state_dict)

# #%%
# with open('11-pretrained_incremental_learning/data/Train_vehicle.pkl', 'rb') as file:
#     Train_vehicle = pickle.load(file)
#
# with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
#     Test_vehicle = pickle.load(file)
#
# with open('11-pretrained_incremental_learning/normalization_params.pkl', 'rb') as f:
#     min_values, max_values = pickle.load(f)
#
#
# train_all_simu = []
# train_all_real = []
# train_M=[]
# for arr in Train_vehicle:
#
#     if len(arr) <= 1:
#         continue
#
#     Array=labeling(arr)
#     Normalized_array = (Array - min_values) / (max_values - min_values)
#
#     testX = torch.Tensor(Normalized_array.iloc[:, 1:].values).float()
#     testY = torch.Tensor(Normalized_array.iloc[:, 0].values).float()
#
#     GPECM1.to(device).eval()
#     with torch.no_grad():
#         pred, _ = GPECM1(testX.to(device))
#         predictions = pred.data.cpu().numpy()
#         targets=testY.cpu().numpy()
#
#     EC_True = ((targets) * (max_values[0] - min_values[0]) / 2) + min_values[0]
#     EC_Pred = ((np.abs(predictions)) * (max_values[0] - min_values[0]) / 2) + min_values[0]
#     Metric1 = np.array(evaluation(EC_True, EC_Pred))
#     print("acc:", Metric1)
#     train_M.append(Metric1)
#     train_all_simu.append(EC_Pred.reshape(-1,1))
#     train_all_real.append(EC_True.reshape(-1,1))
#
#
# M1_train_ind=np.mean(np.vstack(train_M),axis=0)
# M1_train_all=np.array(evaluation(np.vstack(train_all_real),
#                                  np.vstack(train_all_simu)))
#
#
# test_M=[]
# test_all_simu = []
# test_all_real = []
# for arr in Test_vehicle:
#
#     Array=labeling(arr)
#     Normalized_array = (Array - min_values) / (max_values - min_values)
#
#     testX = torch.Tensor(Normalized_array.iloc[:, 1:].values).float()
#     testY = torch.Tensor(Normalized_array.iloc[:, 0].values).float()
#
#     GPECM1.to(device).eval()
#     with torch.no_grad():
#         pred, _ = GPECM1(testX.to(device))
#         predictions = pred.data.cpu().numpy()
#         targets=testY.cpu().numpy()
#
#     EC_True = ((targets) * (max_values[0] - min_values[0]) / 2) + min_values[0]
#     EC_Pred = ((np.abs(predictions)) * (max_values[0] - min_values[0]) / 2) + min_values[0]
#     Metric1 = np.array(evaluation(EC_True, EC_Pred))
#     print("acc:", Metric1)
#     test_M.append(Metric1)
#     test_all_simu.append(EC_Pred.reshape(-1,1))
#     test_all_real.append(EC_True.reshape(-1,1))
#
# M1_test_ind=np.vstack(test_M)
# M1_test_all=np.array(evaluation(np.vstack(test_all_real),
#                                  np.vstack(test_all_simu)))

