# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

#%%
# 数据
evs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
mae_gap = [0.04729, 0.05965, 0.62989, 0.75897, 0.91649, 1.23628, 1.78178, 1.79331, 1.86179, 2.09041, 2.1991, 2.29667, 2.33786, 2.36598, 2.44349, 2.93272, 2.94192, 3.15884, 3.16654, 3.22286]
train_time_gap = [0.01335, 0.22689, 0.51254, 0.35206, 0.03575, 0.16895, 0.23518, 0.02748, 0.523, 0.35765, 0.4797, 0.02582, 0.08929, 0.491, 0.20424, 0.37791, 0.0211, 0.1845, 0.02721, 0.37271]
infer_time_gap = [0, -0.00293, -6E-05, 0.0028, 0, 0.00081, 0, 0, 0.00179, 0, -0.00059, 0, 0, 0.00315, -0.00029, -0.00074, 0, -0.00076, -0.00001, -0.00108]

# 创建图形和子图
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax1 = plt.subplots(figsize=(12,5))

# 设置第一个Y轴 - MAE GAP的柱状图
bar = ax1.bar(evs, mae_gap, color='#427AB2', alpha=0.6, label='MAE')
ax1.set_xlabel('EVs (Ordered by MAE )', fontsize=18)
ax1.set_ylabel('MAE GAP', fontsize=18, color='#427AB2')
ax1.tick_params(axis='y', labelcolor='#427AB2')

# 设置第二个Y轴 - Train time gap的折线图
ax2 = ax1.twinx()
line1, = ax2.plot(evs, train_time_gap, color='#343434', marker='o', linestyle='--', label='Train time')
ax2.set_ylabel('Train Time Gap (s)', fontsize=18, color='#343434')
ax2.tick_params(axis='y', labelcolor='#343434')

# 设置第三个Y轴 - Infer time gap的折线图
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # 调整第三个Y轴的位置
line2, = ax3.plot(evs, infer_time_gap, color='#C74647', marker='s', linestyle='--', label='Infer time')
ax3.set_ylabel('Infer Time Gap (s)', fontsize=18, color='#C74647')
ax3.tick_params(axis='y', labelcolor='#C74647')

# 设置x轴和y轴的刻度字体大小
ax1.set_xticklabels(evs, fontsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)
ax3.tick_params(axis='y', labelsize=18)

# 添加图例：将每个图形对象传递给 legend
lines = [bar, line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels,ncol=3, loc='upper left', fontsize=20)

# 图例
fig.tight_layout()
plt.savefig(r"11-pretrained_incremental_learning/Fig/individual.svg", dpi=600)
plt.show()