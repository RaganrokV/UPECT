# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import math
import torch.nn.parallel
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
#%%


"""Note that shap is not able to interpret UPECT directly. 
therefore, we have no choice but to compute explainer-UPECT first via shap.DeepExplainer. 
then, train an xgb and replace xgb-UPECT with explainer-UPECT to achieve interpretability."""


"""In addition, SHAP seems to be extremely time-consuming to interpret for large models, 
and we were unable to compute feature importance for UPECT"""

#%%
with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

Array=pd.concat(Test_vehicle , ignore_index=True)

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

def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())

# # 对每一列进行归一化操作
Normalized_array = Array.apply(min_max_normalize, axis=0)
del Array



x = Normalized_array.values[:, 1:]
y = Normalized_array.values[:, 0]



#%%
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

        for param in self.parameters():
            param.requires_grad = False

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

        return x


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

UPECT = Transformer_encoder().to(device)
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

UPECT.load_state_dict(new_state_dict)
UPECT.eval()
del ckpt, state_dict,new_state_dict
#%%


idx=5
with open('11-pretrained_incremental_learning/data/testX.pkl', 'rb') as f:
    TestX = pickle.load(f)

# 创建DeepExplainer实例
explainer = shap.DeepExplainer(UPECT, TestX[:1000,:].to(device))
# 释放内存资源
del UPECT

shap_values = explainer.shap_values(TestX[idx-1:idx,:].to(device))

# 定义梯度提升树的参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    # 其他参数也可以在此添加
}

# 初始化梯度提升树回归器
xgb_regressor = xgb.XGBRegressor(random_state=42)

# 在内部循环中进行网格搜索
grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=2)
grid_search.fit(x[:1000], y[:1000])

# 使用最佳参数的模型进行预测
best_xgb_regressor = grid_search.best_estimator_
explainer = shap.Explainer(best_xgb_regressor, x[:1000,:])
shap_values2 = explainer(x)
shap_values3=shap_values2[idx]
shap_values3.values=shap_values.flatten()
#%%
plt.rcParams['font.family'] = 'Times New Roman'
# fig, ax = plt.subplots(figsize=(15, 4))

legend_labels=['Duration', 'Distance', 'Speed', 'State of charge', 'Odometer readings',
               'Battery voltage range', 'Battery temperature range', 'Insulation resistance', 'Temperature',  'Pressure', 'Humidity',
               'Wind speed','Visibility', 'Precipitation', 'Season', 'Date', 'Period', 'Battery capacity', 'Vehicle Type', 'Curb Weight']
shap_values3.feature_names=legend_labels
plt.rcParams['font.family'] = 'Times New Roman'

fig=shap.plots.waterfall(shap_values3,show=False)
fig.set_size_inches(8, 5)
# fig.set_dpi(600)
plt.tight_layout()
plt.savefig(r"11-pretrained_incremental_learning/Fig/Explanation.svg", dpi=600)
plt.show()






