# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd
import pickle
import torch
import warnings
warnings.filterwarnings("ignore")
#%%

FR1 = torch.load("11-pretrained_incremental_learning/model/Feat_Representation_1000.pt")
FR2 = torch.load("11-pretrained_incremental_learning/model/Feat_Representation_2000.pt")
FR3 = torch.load("11-pretrained_incremental_learning/model/Feat_Representation_3000.pt")
FR4 = torch.load("11-pretrained_incremental_learning/model/Feat_Representation_4000.pt")
FR5 = torch.load("11-pretrained_incremental_learning/model/Feat_Representation_5000.pt")
FRs = [FR1.detach().cpu().numpy(), FR2.detach().cpu().numpy(), FR3.detach().cpu().numpy(),
       FR4.detach().cpu().numpy(), FR5.detach().cpu().numpy()]
#%%
# 计算每个特征在512个维度上的均值
means = [np.mean(FR, axis=1) for FR in FRs]

M=np.vstack(means)

plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(8, 6))

unique_colors = plt.cm.tab20b(np.linspace(0, 1, 20))


legend_labels=['Duration', 'Distance', 'Speed', 'State of charge', 'Odometer readings',
               'Battery voltage range', 'Battery temperature range', 'Insulation resistance', 'Temperature',  'Pressure', 'Humidity',
               'Wind speed','Visibility', 'Precipitation', 'Season', 'Date', 'Period', 'Battery capacity', 'Vehicle Type', 'Curb Weight']
# 绘制曲线
for i in range(M.shape[1]):
    plt.plot(range(1, len(FRs)+1), M[:, i], marker='o',
             color=unique_colors[i],
             label=legend_labels[i])
# 添加图例、标题、标签等
plt.legend(ncol=1, loc='center',
           bbox_to_anchor=(1.35, 0.5),
           fontsize=12
           )

plt.xlabel('Pretraining epochs', fontsize='15')
plt.ylabel('Mean feature representation', fontsize='15')

# 显示图形
plt.grid(True)
plt.xticks(range(1, len(FRs)+1), [1000, 2000, 3000, 4000, 5000],
           fontsize='15')
plt.yticks(fontsize='15')
plt.tight_layout()
plt.savefig(r"11-pretrained_incremental_learning/Fig/Feat_Representation.svg", dpi=600)
plt.show()
