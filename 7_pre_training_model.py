# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.utils.data as Data
from thop import profile
from thop import clever_format
import math
import torch.nn.parallel
import os
#%% load

with open('Pre-training_EC/Train_vehicle.pkl', 'rb') as file:
    Train_vehicle = pickle.load(file)

Array=pd.concat(Train_vehicle, ignore_index=True)
#%%
"""label encoding"""

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

columns_to_drop = ["出行时间", "地点", "VIN"]
Array = Array.drop(columns=columns_to_drop).astype(float)
Array = Array.fillna(0)
#%%
Shuffled_Array = Array.sample(frac=1, random_state=42).reset_index(drop=True)
del Array

min_values = Shuffled_Array.min()
max_values = Shuffled_Array.max()

with open('11-pretrained_incremental_learning/normalization_params.pkl', 'wb') as f:
    pickle.dump((min_values, max_values), f)


def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())

# 对每一列进行归一化操作
Normalized_array = Shuffled_Array.apply(min_max_normalize, axis=0)
del Shuffled_Array

trainX = torch.Tensor(Normalized_array.iloc[:, 1:].values).float()
trainY = torch.Tensor(Normalized_array.iloc[:, 0].values).float()

train_dataset = Data.TensorDataset(trainX, trainY)
Dataloaders_train = Data.DataLoader(dataset=train_dataset,
                                    batch_size=128, shuffle=True,
                                    generator=torch.Generator().manual_seed(42))

# del Normalized_array,trainX,trainY,train_dataset
#%%
# class Transformer_src2tgt(nn.Module):
#     def __init__(self):
#         super(Transformer_src2tgt, self).__init__()
#
#         self.Feat_embedding = nn.Linear(1, 512, bias=False) # equal to nn.embedding
#         self.src2tgt = nn.Linear(20, 1, bias=True)
#         self.pos = PositionalEncoding(512,max_len=100)
#         self.transform = nn.Transformer(
#             d_model=512,
#             nhead=8,
#             num_encoder_layers=6,
#             num_decoder_layers=6,
#             dim_feedforward=2048,
#             batch_first=True,
#             dropout=0.5,
#             activation="gelu",
#         )
#         self.out_fc = nn.Linear(512, 1, bias=True)
#         self.activation = nn.ReLU()  # 添加激活函数
#         self.dropout = nn.Dropout(0.1)  # 添加Dropout层
#         self.bn = nn.BatchNorm1d(1)
#
#
#     def forward(self, src):
#         # embedding
#         embedding_src=self.Feat_embedding(src.unsqueeze(2))#(128,20,1)--(128,20,512)
#
#         tgt=self.src2tgt(src)
#
#         embedding_tgt = self.Feat_embedding(tgt.unsqueeze(2))
#
#         embed_encoder_input = self.pos(embedding_src) #essential add a shift
#         embed_decoder_input = self.pos(embedding_tgt)  # essential add a shift
#
#         # transform
#         out = self.transform(embed_encoder_input, embed_decoder_input) #(128,1,512)
#
#         x = self.bn(out)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.out_fc(x)
#
#
#         return x,embedding_src


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
# Initialize your Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 初始化多个 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定可见的 GPU

PECM = Transformer_encoder()
# PECM = Transformer_src2tgt()


PECM.to(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    PECM = nn.DataParallel(PECM)  # 使用DataParallel来包装模型
else:
    print("Let's use single GPU!")

PECM.apply(initialize_weights)

optimizer = torch.optim.AdamW(PECM.parameters(), lr=1e-5,
                              betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       patience=10, factor=0.99)
criterion = nn.MSELoss()

#%%

# 设置初始的 epoch 和 total_loss
start_epoch = 0


# 检查点路径和保存间隔
models_folder = '11-pretrained_incremental_learning/model'
os.makedirs(models_folder, exist_ok=True)
checkpoint_path = os.path.join(models_folder, 'model_checkpoint.pt')
save_interval = 1000  # 每训练5000步保存一次模型

# 检查是否存在之前保存的检查点
if os.path.exists(checkpoint_path):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    if 'epoch' in checkpoint and 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        PECM.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_loss = checkpoint['total_loss']


log_interval = int(len(Dataloaders_train.dataset) / 128 / 5)


for epoch in range(start_epoch, 300000):
    ES = []
    step = 0
    total_loss = 0.0
    for step, (x, y) in enumerate(Dataloaders_train):
        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        pre_y, embedding_src = PECM(x)

        ES.append(torch.mean(embedding_src, dim=0))

        loss = criterion(pre_y, y.unsqueeze(1))

        loss.backward()

        gradients = []
        for param in PECM.parameters():
            if param.grad is not None:
                gradients.append(param.grad.norm().item())

        #         if step % 1000 == 0 and step > 0:
        #             # 打印或保存梯度的变化情况
        #             print("Gradients:", gradients)

        torch.nn.utils.clip_grad_norm_(PECM.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0 and (step + 1) > 0:
            cur_loss = total_loss / log_interval
            mean_gradient = torch.mean(torch.tensor(gradients))
            print('| epoch {:3d} | {:5d}/{:5d} batches | ''lr {:02.6f} | ''loss {:5.5f} | ''mean gradient {:5.5f}'
                  .format(epoch, (step + 1), len(Dataloaders_train.dataset) // 128,
                          optimizer.param_groups[0]['lr'], cur_loss, mean_gradient))

            total_loss = 0

    scheduler.step(total_loss)

    Feat_Representation = torch.stack(ES).mean(0)

    del ES
    torch.cuda.empty_cache()

    if (epoch + 1) % 5 == 0:
        print('-' * 89)
        print(
            f"Epoch {epoch + 1}: Learning Rate: {optimizer.param_groups[0]['lr']:.9f}, Loss: {total_loss / len(Dataloaders_train):.9f}")
        print('-' * 89)

    # 保存模型和特征表示
    if (epoch + 1) % save_interval == 0:
        model_path = 'P_{}'.format(epoch + 1)
        torch.save(PECM.state_dict(), model_path)
        feat_path = 'Feat_Representation_{}.pt'.format(epoch + 1)
        torch.save(Feat_Representation, feat_path)

    # 保存检查点
    if (epoch + 1) % 1000 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': PECM.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_loss': total_loss,
            'Feat_Representation': Feat_Representation
        }
        torch.save(checkpoint, checkpoint_path)

#%%
# Initialize your Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

PECM = Transformer_src2tgt()
ckpt = torch.load('11-pretrained_incremental_learning/model/model_checkpoint_BN512.pt')

# 提取 DataParallel 包装的模型中的实际模型参数
state_dict = ckpt["model_state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_key = key[7:]  # 去除 'module.' 的前缀
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

PECM.load_state_dict(new_state_dict)

from My_utils.evaluation_scheme import evaluation
PECM.to(device).eval()
all_simu = []
all_real = []
for i, (x, y) in enumerate(Dataloaders_train):
    with torch.no_grad():
        pred, _ = PECM(x.to(device))
        Norm_pred = pred.data.cpu().numpy()
        all_simu.append(Norm_pred)
        all_real.append(y.unsqueeze(1).numpy())

targets=np.vstack(all_real)
predictions=np.vstack(all_simu)


EC_True=((targets) * (max_values[0] - min_values[0]) / 2) + min_values[0]
EC_Pred=((predictions) * (max_values[0] - min_values[0]) / 2) + min_values[0]


Metric1=np.array(evaluation(EC_True, EC_Pred))
print("acc:",Metric1)




#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(PECM):,} trainable parameters')


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

getModelSize(PECM)
#%%
# 估算 FLOPs
flops, params = profile(PECM, inputs=(x.to(device),))

# 格式化输出
flops, params = clever_format([flops, params], "%.3f")

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")

