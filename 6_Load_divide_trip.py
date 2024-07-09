# -*- coding: utf-8 -*-
import pandas as pd
import os
import warnings
import pickle
warnings.filterwarnings("ignore")
import glob
import random
#%%
# 指定存储.pkl文件的文件夹路径
folder_path = '11-pretrained_incremental_learning/data'

# 使用glob.glob()获取文件夹中所有的.pkl文件
pkl_files = glob.glob(os.path.join(folder_path, '*.pkl'))

# 遍历找到的文件列表
for file_path in pkl_files:
    # 使用os.path.basename获取完整的文件名（包括扩展名）
    file_name_with_extension = os.path.basename(file_path)

    variable_name = file_name_with_extension.replace('.pkl', '').replace('.', '_')

    # 读取.pkl文件
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 使用处理后的文件名作为变量名，将数据保存到全局变量中
    globals()[variable_name] = data

#%%
All_trip=CND01Y22+SHD01Y21+SHD02Y22M1+SHD02Y22M2+SHD02Y22M3+SHD02Y22M4+SHD02Y22M5+SHD02Y22M6+SHD03Y21+SZD01Y22
All_trip_df = pd.concat(All_trip, ignore_index=True)
#%% test dataset

Test_vehicle=[CND01Y22[3],CND01Y22[8]]+[SHD01Y21[0],SHD01Y21[15]]+[SHD02Y22M1[5],SHD02Y22M1[10]]+\
             [SHD02Y22M2[5],SHD02Y22M2[21]]+[SHD02Y22M3[7],SHD02Y22M3[9]]+[SHD02Y22M4[13],SHD02Y22M4[42],SHD02Y22M4[52]]+\
             [SHD02Y22M6[16],SHD02Y22M6[26]]+[SHD03Y21[3],SHD03Y21[17],SHD03Y21[64],SHD03Y21[96]]+[SZD01Y22[6]]

A= pd.concat(Test_vehicle, ignore_index=True)
#%% train dataset
Train_vehicle=[item for index, item in enumerate(CND01Y22) if index not in [3, 8]]+\
              [item for index, item in enumerate(SHD01Y21) if index not in [0, 15]]+\
              [item for index, item in enumerate(SHD02Y22M1) if index not in [5, 10]]+\
              [item for index, item in enumerate(SHD02Y22M2) if index not in [5, 21]]+\
              [item for index, item in enumerate(SHD02Y22M3) if index not in [7, 9]]+\
              [item for index, item in enumerate(SHD02Y22M4) if index not in [13, 42,52]]+\
              SHD02Y22M5+\
              [item for index, item in enumerate(SHD02Y22M6) if index not in [16, 26]]+\
              [item for index, item in enumerate(SHD03Y21) if index not in [3, 17,64,96]]+\
              [item for index, item in enumerate(SZD01Y22) if index not in [6]]
B= pd.concat(Train_vehicle, ignore_index=True)
#%% enhance

# 遍历df_list中的每个df，对每个df进行数据增强
enhanced_Train_vehicle=[]
for df in Train_vehicle:
    if len(df) > 50:
        # 随机选择20行
        rows_to_augment = random.sample(range(len(df)), 20)

        # 复制选中的行并将它们添加到原始df中
        for row_idx in rows_to_augment:
            duplicated_row = df.iloc[row_idx].copy()
            df = pd.concat([df, duplicated_row.to_frame().T], ignore_index=True)

        # 增加0-5%的噪声到第0，2，3，4列
        for row_idx in rows_to_augment:
            for col in [0, 2, 3, 4]:
                noise = random.uniform(-0.05, 0.05)  # 生成-5%到5%之间的随机数
                df.iloc[row_idx, col] = df.iloc[row_idx, col] * (1 + noise)  # 对指定列应用声

    enhanced_Train_vehicle.append(df)

C = pd.concat(enhanced_Train_vehicle, ignore_index=True)

#%% save
#trian 30412; test 13221
with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'wb') as file:
    pickle.dump(Test_vehicle, file)

with open('11-pretrained_incremental_learning/data/Train_vehicle.pkl', 'wb') as file:
    pickle.dump(enhanced_Train_vehicle, file)
