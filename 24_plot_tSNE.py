# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd
import pickle
#%%
with open('11-pretrained_incremental_learning/data/Embeddings.pkl', 'rb') as f:
    loaded_Embeddings = pickle.load(f)

with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)
#%%
def visual_3d(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=3, perplexity=50)

    x_ts = ts.fit_transform(feat)

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final

vehicle_mapping = {'Sedan': 0, 'SUV': 1, 'Sedan PHEV': 2, 'SUV PHEV': 2}
types = pd.concat(Test_vehicle, ignore_index=True)["车辆类型"].map(vehicle_mapping).values

true_labels = types.reshape((-1, 1))

hdr_3d = visual_3d(loaded_Embeddings)

s_data_3d = np.hstack((hdr_3d, true_labels))  # 将降维后的特征与相应的标签拼接在一起

s_data_3d = pd.DataFrame({'x': s_data_3d[:, 0], 'y': s_data_3d[:, 1], 'z': s_data_3d[:, 2],
                          'label': s_data_3d[:, 3]})
#%%
# 创建绘图对象
plt.rcParams['font.family'] = 'Times New Roman'
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 定义颜色映射
colors = ['#2878B5', '#C82423',
          'g']
labels = ['BEV Sedan', 'BEV SUV', 'PHEV']

# 遍历每种车辆类型，绘制对应颜色的3D散点图
for i, label in enumerate(labels):
    subset = s_data_3d[s_data_3d['label'] == i]
    ax.scatter(subset['x'], subset['y'], subset['z'],
               c=colors[i],
               # marker=markers[i],
               s=1,
               alpha=0.6,
               label=label)

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# 添加图例
legend = ax.legend(prop={'size': 18},
                   loc='upper right',
                   bbox_to_anchor=(0.95, 0.85)
                   , scatterpoints=25,
                    frameon=False)
ax.grid(None)

ax.axis('off')
# 显示轴，但不显示刻度线
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
ax.view_init(elev=20, azim=35)
# ax.invert_xaxis()
# ax.invert_yaxis()
ax.plot([0, 0.7], [0, 0], [0, 0], color='k', linewidth=0.8)
# Y 轴
ax.plot([0, 0], [0, 0.8], [0, 0], color='k', linewidth=0.8)
# Z 轴
ax.plot([0, 0], [0, 0], [0, 0.9], color='k', linewidth=0.8)
ax.text(0.7, 0, -0.07, 'Dimension 1', color='k', fontsize=12, ha='center')
ax.text(0, 0.82, -0.07, 'Dimension 2', color='k', fontsize=12, ha='center')
ax.text(0, 0, 0.95, 'Dimension 3', color='k', fontsize=12, ha='center')
plt.tight_layout()
# 显示图形
plt.savefig(r"11-pretrained_incremental_learning/Fig/tSNE_3D.svg", dpi=600)
plt.show()
#%%

def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2,perplexity=50,)

    x_ts = ts.fit_transform(feat)

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final

vehicle_mapping = {'Sedan': 0, 'SUV': 1, 'Sedan PHEV': 2, 'SUV PHEV': 2}
type = pd.concat(Test_vehicle, ignore_index=True)["车辆类型"].map(vehicle_mapping).values

True_labels = type.reshape((-1, 1))

HDR=visual(loaded_Embeddings)

S_data = np.hstack((HDR, True_labels))  # 将降维后的特征与相应的标签拼接在一起

S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1],'z': S_data[:, 1],
                       'label': S_data[:, 2]})

#%%
# 定义颜色映射
colors = ['r', 'g', 'b']
labels = ['Sedan', 'SUV', 'PHEV']

# 创建绘图对象
plt.figure(figsize=(8, 6))

# 遍历每种车辆类型，绘制对应颜色的散点图
for i, label in enumerate(labels):
    subset = S_data[S_data['label'] == i]
    plt.scatter(subset['x'], subset['y'], c=colors[i], label=label)

# 添加图例
plt.legend()

# 设置图形标题和轴标签
plt.title('t-SNE Visualization of Vehicle Types')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# 显示图形
plt.show()
#%%
fig = plt.figure(figsize=(10, 10))
models=['Sedan', 'SUV', 'PHEV']
for index in range(3):  #
    X = S_data.loc[S_data['label'] == index]['x']
    Y = S_data.loc[S_data['label'] == index]['y']
    plt.scatter(X, Y, cmap='viridis', s=15,
                # marker=maker[index],
                # c=colors[index], edgecolors=colors[index],
                alpha=0.65,
                label=models[index])

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.legend(fontsize=15)


plt.show()



