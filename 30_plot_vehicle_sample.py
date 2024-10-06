# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid.inset_locator import inset_axes, mark_inset
from matplotlib.lines import Line2D

#%%
with open('11-pretrained_incremental_learning/data_plot/ECR_data.pkl', 'rb') as f:
    data = pickle.load(f)

ECR_real = data['EC_True'][-573:,]
ECR_pred = data['EC_Pred'][-573:,]

with open('11-pretrained_incremental_learning/data_plot/GPT_data.pkl', 'rb') as f:
    data = pickle.load(f)

GPR_real = data['EC_True']
GPR_pred = data['EC_Pred']

with open('11-pretrained_incremental_learning/data_plot/RF_data.pkl', 'rb') as f:
    data = pickle.load(f)

RF_real = data['EC_True']
RF_pred = data['EC_Pred']

with open('11-pretrained_incremental_learning/data_plot/XGB_data.pkl', 'rb') as f:
    data = pickle.load(f)

XGB_real = data['EC_True']
XGB_pred = data['EC_Pred']

with open('11-pretrained_incremental_learning/data_plot/UPECT_data.pkl', 'rb') as f:
    data = pickle.load(f)

UPECT_real = data['EC_True']
UPECT_pred = data['EC_Pred']
#%%
plt.rcParams['font.family'] = 'Times New Roman'
# 颜色定义
colors = ['#C74647', '#C59D94', '#F2BB6B',
          '#D3D3D3', '#CDE8C3', '#427AB2']

# 创建主图
fig, ax = plt.subplots(figsize=(15, 6))

# 绘制不同模型的预测值和真实值
plt.plot(ECR_pred, label='ECR', marker='o', color=colors[1], lw=0.5)
plt.plot(GPR_pred, label='GPR', marker='s', color=colors[2], lw=0.5)
plt.plot(RF_pred, label='RF', marker='^', color=colors[3], lw=0.5)
plt.plot(XGB_pred, label='XGB', marker='x', color=colors[4], lw=0.5)
plt.plot(UPECT_pred, label='UPECT', marker='d', color=colors[5], lw=0.5)
plt.plot(UPECT_real, label='Real', linestyle='--', color=colors[0], lw=1)

# 创建自定义图例
legend_lines = [
    Line2D([0], [0], color=colors[1], marker='o', linestyle='-', lw=2.5, markersize=8, label='ECR'),
    Line2D([0], [0], color=colors[2], marker='s', linestyle='-', lw=2.5, markersize=8, label='GPR'),
    Line2D([0], [0], color=colors[3], marker='^', linestyle='-', lw=2.5, markersize=8, label='RF'),
    Line2D([0], [0], color=colors[4], marker='x', linestyle='-', lw=2.5, markersize=8, label='XGB'),
    Line2D([0], [0], color=colors[5], marker='d', linestyle='-', lw=2.5, markersize=8, label='UPECT'),
    Line2D([0], [0], color=colors[0], linestyle='--', lw=2.5, label='Real')
]

plt.xlabel('Trips over time', fontsize=25)
plt.ylabel('Energy consumption (kWh)', fontsize=25)

plt.ylim(-1, 30)    # 设置y轴范围
plt.grid(True)

# 设置 x 和 y 轴刻度标签大小
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(handles=legend_lines, fontsize=21, ncol=2,loc='upper left')
# 设置放大区域
x1, x2, y1, y2 = 90, 110, -1, 18

# 创建插图区域
axins = inset_axes(ax, width="50%", height="40%", loc='upper right')

# 绘制插图区域
axins.plot(UPECT_real, label='Real', linestyle='--', color=colors[0],
           lw=2.5)
axins.plot(ECR_pred, label='ECR', marker='o', color=colors[1],
           lw=.5,alpha=0.7)
axins.plot(GPR_pred, label='GPR', marker='s', color=colors[2],
           lw=.5,alpha=0.7)
axins.plot(RF_pred, label='RF', marker='^', color=colors[3],
           lw=.5,alpha=0.7)
axins.plot(XGB_pred, label='XGB', marker='x', color=colors[4],
           lw=.5,alpha=0.7)
axins.plot(UPECT_pred, label='UPECT', marker='d', color=colors[5],
           lw=2.5)

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticks([])
axins.set_yticks([])

# 在主图上标记插图的位置
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='b', lw=3, linestyle='--')

# 绘制红色矩形框突出显示放大区域
rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=6, edgecolor='b', facecolor='none', linestyle='-')
ax.add_patch(rect)



plt.tight_layout()
plt.savefig('11-pretrained_incremental_learning/Fig/sample.svg', dpi=600)
plt.savefig('11-pretrained_incremental_learning/Fig/sample.png', dpi=600)
plt.show()

