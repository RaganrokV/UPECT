# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt



#%%
# 读取数据的函数
plt.rcParams['font.family'] = 'Times New Roman'

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 定义数据集和模型的顺序
datasets = ['ChargeCar','VED', 'SpritMonitor', 'BHD', ]
models = [ 'GPR', 'RF', 'XGB', 'UPECT',]

metrics = {
    'ChargeCar': [
        round(1.48663, 3), round(-0.01304, 2),
        round(1.78248, 3), round(0.22358, 2),
        round(1.65647, 3), round(0.29866, 2),
        round(0.64649, 3), round(0.45855, 2),

    ],
    'BHD': [
        round(1.43181, 3), round(-0.26038, 2),
        round(1.27456, 3), round(-0.02467, 2),
        round(0.95094, 3), round(0.20147, 2),
        round(0.5095, 3), round(0.09555, 2),

    ],
    'SpritMonitor': [
        round(3.73619, 3), round(0.08739, 2),
        round(1.59455, 3), round(0.74232, 2),
        round(3.19443, 3), round(0.26654, 2),
        round(1.75653, 3), round(0.14823, 2),

    ],
    'VED': [
        round(0.46878, 3), round(-0.11131, 2),
        round(0.32766, 3), round(0.43906, 2),
        round(0.26926, 3), round(0.44932, 2),
        round(0.19176, 3), round(0.23038, 2),

    ]
}

# 每个文件的路径模板
file_template = '11-pretrained_incremental_learning/data_plot/{}_{}.pkl'

# 创建子图
fig, axs = plt.subplots(len(models), len(datasets), figsize=(9, 12))

# 定义字体大小
label_fontsize = 14
tick_fontsize = 12
text_fontsize = 10

# 遍历模型和数据集，绘制散点图
for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        file_path = file_template.format(model, dataset)

        try:
            data = load_data(file_path)
            real = data['EC_True']
            pred = data['EC_Pred']

            axs[i, j].scatter(real, pred, alpha=0.6,c='#407BD0',s=3)
            axs[i, j].set_xlabel('True', fontsize=label_fontsize)
            axs[i, j].set_ylabel('Predicted', fontsize=label_fontsize)
            axs[i, j].plot([min(real), max(real)], [min(real), max(real)], color='#A32A31', linestyle='--')  # 参考线

            # 设置刻度字体大小
            axs[i, j].tick_params(axis='both', labelsize=tick_fontsize)

            # 添加MAE和R²值到左上角
            mae = metrics[dataset][i * 2] if metrics[dataset][i * 2] is not None else 'N/A'
            r2 = metrics[dataset][i * 2 + 1] if metrics[dataset][i * 2 + 1] is not None else 'N/A'
            axs[i, j].text(0.05, 0.95, f'MAE: {mae}\nR²: {r2}',
                           ha='left', va='top', transform=axs[i, j].transAxes,
                           fontsize=15)

            # 在每个子图顶部添加数据集名称
            if i == 0:  # 只在第一行添加
                axs[i, j].text(0.5, 1.05, dataset, ha='center', va='bottom',
                               transform=axs[i, j].transAxes,
                               fontsize=20)

            # 在每个子图左侧添加模型名称
            if j == 0:  # 只在第一列添加
                axs[i, j].text(-0.4, 0.5, model, ha='right', va='center',
                               transform=axs[i, j].transAxes, rotation=90,
                               fontsize=20)

        except FileNotFoundError:
            axs[i, j].set_visible(False)  # 如果文件不存在，则隐藏该子图

# 调整布局
plt.tight_layout()
plt.savefig('11-pretrained_incremental_learning/Fig/scatter.svg', dpi=600)
plt.show()