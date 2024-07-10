# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import warnings
import pickle
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

from thop import profile
from thop import clever_format
import math
import torch.nn.parallel
import os
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

UPECT = Transformer_encoder()


#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(UPECT):,} trainable parameters')

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

getModelSize(UPECT)
#%%


# 估算 FLOPs
flops, params = profile(UPECT, inputs=(x.to(device),))

# 格式化输出
flops, params = clever_format([flops, params], "%.3f")

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")



