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
import torch.utils.data as Data
import time
#%% load
# 读取 SpritMonitor.pkl 中的数据
with open("11-pretrained_incremental_learning/data/SpritMonitor.pkl", "rb") as f:
    SpritMonitor = pickle.load(f)

#%%
"""generalization"""
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


"""V1"""
V1_min_value=SpritMonitor_labeled[0]['行程能耗'].min()
V1_max_value=SpritMonitor_labeled[0]['行程能耗'].max()

Internal_normalize1 = ['行程能耗', '行程距离', '当前累积行驶里程']

# 找到每列的最大最小值，并进行归一化
SpritMonitor_labeled[0][Internal_normalize1] = (SpritMonitor_labeled[0][Internal_normalize1] -
                                                 SpritMonitor_labeled[0][Internal_normalize1].min())\
                                                / (SpritMonitor_labeled[0][Internal_normalize1].max() -
                                                SpritMonitor_labeled[0][Internal_normalize1].min())


# for column in Internal_normalize1:
#     min_val = min_values[column]
#     max_val = max_values[column]
#     SpritMonitor_labeled[0][column] = np.abs(SpritMonitor_labeled[0][column] - min_val) / (max_val - min_val)

"""V2"""
V2_min_value=SpritMonitor_labeled[1]['行程能耗'].min()
V2_max_value=SpritMonitor_labeled[1]['行程能耗'].max()

Internal_normalize2 = ['行程能耗', '行程距离' ,"行程平均速度"]

SpritMonitor_labeled[1][Internal_normalize2] = (SpritMonitor_labeled[1][Internal_normalize2] -
                                                 SpritMonitor_labeled[1][Internal_normalize2].min())\
                                                / (SpritMonitor_labeled[1][Internal_normalize2].max() -
                                                SpritMonitor_labeled[1][Internal_normalize2].min())

# for column in Internal_normalize2:
#     min_val = min_values[column]
#     max_val = max_values[column]
#     SpritMonitor_labeled[1][column] =  np.abs(SpritMonitor_labeled[1][column] - min_val) / (max_val - min_val)
#


# 创建一个默认值字典
feat_name = {'行程能耗': False, '行程时间': False, '行程距离': False,
             '行程平均速度': False, '当前SOC': False, '当前累积行驶里程': False,
             '单体电池电压极差': False, '单体电池温度极差': False, '绝缘电阻值': False,
             'T': False, 'Po': False, 'U': False, 'Ff': False, 'VV': False,
             'RRR': False, '出行季节': False, '出行日期': False, '出行时段': False,
             '电池能量': False, '车辆类型': False, '整备质量': False, 'tire_type': False,
             'city': False, 'motor_way': False, 'country_roads': False,'driving_style': False}

# 将默认值字典转换为DataFrame
default_structure = pd.DataFrame(feat_name, index=[0])
# 提取'行程能耗'列的最大最小值

# 将默认值DataFrame与ChargeCar DataFrame进行连接
SpritMonitor_structured=[]
for dt in SpritMonitor_labeled:
    structured = pd.concat([dt, default_structure],
                                        ignore_index=True).astype(float)
    structured.fillna(value=np.e, inplace=True)
    SpritMonitor_structured.append(structured[feat_name.keys()])
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

        """only for transfer learning, please do not compile this code """
        """1.refreeze the last layer """
        for param in self.parameters():
            param.requires_grad = False

        # 解冻最后的全连接层
        self.out_fc.weight.requires_grad = True
        self.out_fc.bias.requires_grad = True

        # # 解冻最后的全连接层和Feat_embedding
        # for param in self.parameters():
        #     param.requires_grad = True
        #
        # # 冻结transformer_encoder的参数
        # for param in self.transformer_encoder.parameters():
        #     param.requires_grad = False


        """  """

    def forward(self, src,Feat_Representation):

        B,F=src.size()
        Feat_Representation = Feat_Representation.unsqueeze(0).expand(B, -1, -1)

        # embedding
        embedding_src=self.Feat_embedding(src.unsqueeze(2)) #(128,20,1)--(128,20,512)

        e_columns_indices = torch.where(torch.all(src == torch.e, dim=0))[0]

        # 遍历每个特征值都是 torch.e 的列的索引
        for index in e_columns_indices:
            # 使用 Feat_Representation 中对应的列替换 embedding_src 中的数据
            embedding_src[:, index, :] = Feat_Representation[:, index, :]

        embed_encoder_input = self.pos(embedding_src) #essential add a shift

        # transform
        out = self.transformer_encoder(embed_encoder_input) #(128,20,512)

        x = self.bn(out)
        x = self.activation(x)
        x = self.dropout(x)
        # x = self.out_fc(x.reshape(B,-1))


        return x


class General_TransformerEncoder(Transformer_encoder):
    def __init__(self):
        super(General_TransformerEncoder, self).__init__()

        # 添加额外的网络层
        self.activation = nn.ReLU()  # 添加激活函数
        self.dropout = nn.Dropout(0.1)  # 添加Dropout层
        self.bn2 = nn.BatchNorm1d(25)

        self.out_fc_external = nn.Linear(512*5, 1, bias=True)
        self.feat_fusion = nn.Linear(512*25, 1, bias=True)


        # 初始化参数
        # initialize_weights(self.out_fc_external)
        # initialize_weights(self.feat_fusion)

    def forward(self, src, Feat_Representation):

        """independent learning channel"""
        B, F = src.size()

        x = super().forward(src[:,:20], Feat_Representation)

        embedding_src2 = self.Feat_embedding(src[:,20:].unsqueeze(2))

        # fusion_x=embedding_src2
        # # fusion_x=torch.cat((x, embedding_src2), dim=1)
        # # 进行额外的网络层操作
        # # fusion_x = self.out_fc_external(fusion_x.reshape(B,-1))
        #
        # fusion_x = self.bn2(fusion_x)
        # fusion_x = self.activation(fusion_x)
        # fusion_x = self.dropout(fusion_x)
        #
        # # fusion_x=self.feat_fusion(fusion_x.reshape(B,-1))
        # fusion_x = self.out_fc_external(fusion_x.reshape(B, -1))
        #
        # fusion_x=0.7*x+0.3*fusion_x

        fusion_x=torch.cat((x, embedding_src2), dim=1)
        # 进行额外的网络层操作
        # fusion_x = self.out_fc_external(fusion_x.reshape(B,-1))

        fusion_x = self.bn2(fusion_x)
        fusion_x = self.activation(fusion_x)
        fusion_x = self.dropout(fusion_x)

        fusion_x=self.feat_fusion(fusion_x.reshape(B,-1))


        return fusion_x


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
#%%
"""with training"""
for i,structured in enumerate(SpritMonitor_structured):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    UPECT = Transformer_encoder()
    ckpt = torch.load('11-pretrained_incremental_learning/model/UPECT_40M.pt')
    Feat_Representation = ckpt['Feat_Representation']
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

    # 创建新模型实例
    General_model = General_TransformerEncoder()

    # 将预训练模型的参数加载到新模型中
    General_model.load_state_dict(UPECT.state_dict(), strict=False)

    del UPECT

    train_size = int(len(structured) * 0.1)

    trainX = torch.Tensor(structured.iloc[:train_size, 1:].values).float()
    trainY = torch.Tensor(structured.iloc[:train_size, 0].values).float()

    train_dataset = Data.TensorDataset(trainX, trainY)
    Dataloaders_train = Data.DataLoader(dataset=train_dataset,
                                        batch_size=128, shuffle=True,
                                        generator=torch.Generator().manual_seed(42))

    testX = torch.Tensor(structured.iloc[train_size:, 1:].values).float()
    testY = torch.Tensor(structured.iloc[train_size:, 0].values).float()
    optimizer = torch.optim.AdamW(General_model.parameters(), lr=1e-5,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=2, factor=0.99)
    criterion = nn.MSELoss()

    # 记录起始时间
    start_time = time.time()
    num_epochs = 20
    for epoch in range(num_epochs):
        General_model.to(device).train()
        total_loss = 0.
        for step, (x, y) in enumerate(Dataloaders_train):
            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            pre_y = General_model(x,Feat_Representation)

            loss = criterion(pre_y, y.unsqueeze(1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

    end_time = time.time()

    # 计算运行时间
    execution_time = (end_time - start_time) / len(testX)
    print(f"代码运行时间: {execution_time} 秒")

    # 记录起始时间
    start_time = time.time()
    General_model.to(device).eval()
    with torch.no_grad():
        pred = General_model(testX.to(device), Feat_Representation.to(device))
        predictions = pred.data.cpu().numpy()
        targets = testY.cpu().numpy()

    if i==0:
        max_value,min_value=V1_max_value,V1_min_value
    else:
        max_value, min_value = V2_max_value, V2_min_value

    EC_True = ((targets) * (max_value - min_value) / 2) + min_value
    EC_Pred = ((np.abs(predictions)) * (max_value - min_value) / 2) + min_value
    Metric1 = np.array(evaluation(EC_True, EC_Pred))
    print("acc:", Metric1)

    del General_model
    end_time = time.time()

    # 计算运行时间
    execution_time = (end_time - start_time) / len(testX)
    print(f"代码运行时间: {execution_time} 秒")

#%%
"""zero shot"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

UPECT = Transformer_encoder()
ckpt = torch.load('11-pretrained_incremental_learning/model/UPECT_40M.pt')
Feat_Representation = ckpt['Feat_Representation']
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

# 创建新模型实例
General_model = General_TransformerEncoder()

# 将预训练模型的参数加载到新模型中
General_model.load_state_dict(UPECT.state_dict(), strict=False)

testX = torch.Tensor(SpritMonitor_structured[1].iloc[:, 1:].values).float()
testY = torch.Tensor(SpritMonitor_structured[1].iloc[:, 0].values).float()
# 记录起始时间
start_time = time.time()
General_model.to(device).eval()
with torch.no_grad():
    pred = General_model(testX.to(device),Feat_Representation.to(device))
    predictions = pred.data.cpu().numpy()
    targets = testY.cpu().numpy()


# max_value,min_value=V1_max_value,V1_min_value
max_value, min_value = V2_max_value, V2_min_value

EC_True = ((targets) * (max_value - min_value) / 2) + min_value
EC_Pred = ((np.abs(predictions)) * (max_value - min_value) / 2) + min_value
Metric1 = np.array(evaluation(EC_True, EC_Pred))
print("acc:", Metric1)
end_time = time.time()

# 计算运行时间
execution_time = (end_time - start_time) / len(testX)
print(f"代码运行时间: {execution_time} 秒")
