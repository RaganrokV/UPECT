# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns



#%%

plt.rcParams['font.family'] = 'Times New Roman'
models = ['ECR×d', 'GPR', 'RF', 'XGB', 'UPECT']
markers = ['o', 's', '^', 'D', 'h']
colors=["#C74647","#2681B6","#56BA77","#E4B112","#9180AC"]
fig = plt.figure(figsize=(8,6))
percentages = [40, 30, 20, 10, 4]
mae_values = {
    'ECR×d': [2.131245, 2.131245, 2.131245, 2.131245, 2.131245],
    'GPR': [1.535256, 1.888911, 2.251694, 2.701886, None],
    'RF': [1.3212595, 1.415836, 1.701746, 2.1605545, None],
    'XGB': [1.308536, 1.4076005, 1.630168, 2.2440775, None],
    'UPECT': [0.6083, 0.64492, 0.63874, 0.64205, 0.70241]
}

for i, model in enumerate(models):
    plt.plot(percentages, mae_values[model],
             marker=markers[i],markersize=12,
             linestyle='--',color=colors[i],label=model)
plt.xlabel('Training size',fontsize='25')
plt.ylabel('MAE/kWh',fontsize='25')
custom_labels = ['70%', '50%', '30%', '10%', 'Zero shot']
plt.xticks(percentages, custom_labels, fontsize=22)
plt.yticks(fontsize=22)
plt.legend(loc=(0.03, 0.15),fontsize='22')
plt.tight_layout()
# plt.savefig(r"11-pretrained_incremental_learning/Fig/ACC_MAE.svg", dpi=600)
plt.show()

#%%
reduction_results = {}

for method, values in mae_values.items():

    first_value = values[0]
    fourth_value = values[3]

    if first_value == 0:
        reduction = float('inf')  # Handle division by zero if necessary
    else:
        reduction = ((fourth_value-first_value  ) / fourth_value) * 100

    reduction_results[method] = reduction

print("Reduction results:")
for method, reduction in reduction_results.items():
    print(f"{method}: {reduction:.2f}%")
#%%
plt.rcParams['font.family'] = 'Times New Roman'
models = ['ECR×d', 'GPR', 'RF', 'XGB', 'UPECT']
markers = ['o', 's', '^', 'D', 'h']
colors=["#C74647","#2681B6","#56BA77","#E4B112","#9180AC"]
fig = plt.figure(figsize=(8,6))
percentages = [40, 30, 20, 10, 4]
rmse_values = {
    'ECR×d': [3.3238895, 3.3238895, 3.3238895, 3.3238895, 3.3238895],
    'RF': [2.0379035, 2.2304885, 2.55116, 3.199635, None],
    'XGB': [1.972045, 2.2425885, 2.590104, 3.3755135, None],
    'GPR': [2.444056, 3.117864, 3.4533195, 4.1908495, None],
    'UPECT': [1.02331, 1.11077, 1.07361, 1.06687, 1.17592]
}

for i, model in enumerate(models):
    plt.plot(percentages, rmse_values[model],
             marker=markers[i],markersize=12,
             linestyle='--',color=colors[i],label=model)
plt.xlabel('Training size',fontsize='25')
plt.ylabel('RMSE/kWh',fontsize='25')
custom_labels = ['70%', '50%', '30%', '10%', 'Zero shot']
plt.xticks(percentages, custom_labels, fontsize=22)
plt.yticks(fontsize=22)
plt.legend(loc=(0.03, 0.15),fontsize='22')
plt.tight_layout()
plt.savefig(r"11-pretrained_incremental_learning/Fig/ACC_RMSE.svg", dpi=600)
plt.show()