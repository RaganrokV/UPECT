# -*- coding: utf-8 -*-

import pandas as pd
import os
import glob
import pickle

#%%
"""目前总共344，873，979  （3.4亿条）"""
#%%
"SHEV"
V1 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=46U3J2D7.csv')
V2 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=371N6L6H.csv')
V3 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=603F5L4P.csv')
V4 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=5173K555.csv')
V5 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=14704V7Q.csv')
V6 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=33433B5H.csv')
V7 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=73631A6T.csv')
V8 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=K102C490.csv')
V9 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=Q6H131Q0.csv')
V10 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata4137/seqcode=R1V5AT1Q.csv')
V11 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=0A324G2G.csv')
V12 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=1H0A155D.csv')
V13 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=2Q5I5U44.csv')
V14 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=2S5V2C32.csv')
V15 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=3S4Q4G6S.csv')
V16 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=4G353Q4A.csv')
V17 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=4J29166N.csv')
V18 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=6R643A1G.csv')
V19 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=6T3T1S39.csv')
V20 = pd.read_csv('E:\PHEV DATA\EV-SH/batterydata8554/seqcode=7N3I662O.csv')

#%%
"""
    Descrtiption:
    
    上海20辆车，总数据量为：15460365
    时间被打乱需要重新排序
    采样时间从2021-7-30至2022-1-27 约半年
       
"""
lengths = [len(df) for df in [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20]]
total_length = sum(lengths)


for i in range(1, 21):
    var_name = 'V' + str(i)
    globals()[var_name].iloc[:, 0] = pd.to_datetime(globals()[var_name].iloc[:, 0], unit='ms')
    globals()[var_name] = globals()[var_name].sort_values(by=['collectiontime'])

#%%
A1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL1_20220401000000_20220430235959.xlsx')
A2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL1_20220501000000_20220531235959.xlsx')
A3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL1_20220601000000_20220630235959.xlsx')
A4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL1_20220701000000_20220731235959.xlsx')
A5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL1_20220801000000_20220831235959.xlsx')
A6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL1_20220901000000_20220930235959.xlsx')
A7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL1_20221001000000_20221031235959.xlsx')
CAR_A1 = pd.concat([A1, A2, A3, A4, A5, A6, A7])

A1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL2_20220401000000_20220430235959.xlsx')
A2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL2_20220501000000_20220531235959.xlsx')
A3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL2_20220601000000_20220630235959.xlsx')
A4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL2_20220701000000_20220731235959.xlsx')
A5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL2_20220801000000_20220831235959.xlsx')
A6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL2_20220901000000_20220930235959.xlsx')
A7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL2_20221001000000_20221031235959.xlsx')
CAR_A2 = pd.concat([A1, A2, A3, A4, A5, A6, A7])

A1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL3_20220401000000_20220430235959.xlsx')
A2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL3_20220501000000_20220531235959.xlsx')
A3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL3_20220601000000_20220630235959.xlsx')
A4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL3_20220701000000_20220731235959.xlsx')
A5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL3_20220801000000_20220831235959.xlsx')
A6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL3_20220901000000_20220930235959.xlsx')
A7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL3_20221001000000_20221031235959.xlsx')
CAR_A3 = pd.concat([A1, A2, A3, A4, A5, A6, A7])

A1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL4_20220401000000_20220430235959.xlsx')
A2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL4_20220501000000_20220531235959.xlsx')
A3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL4_20220601000000_20220630235959.xlsx')
A4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL4_20220701000000_20220731235959.xlsx')
A5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL4_20220801000000_20220831235959.xlsx')
A6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL4_20220901000000_20220930235959.xlsx')
A7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL4_20221001000000_20221031235959.xlsx')
CAR_A4 = pd.concat([A1, A2, A3, A4, A5, A6, A7])

A1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL5_20220301000000_20220331235959.xlsx')
A2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL5_20220401000000_20220430235959.xlsx')
A3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL5_20220501000000_20220531235959.xlsx')
A4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL5_20220601000000_20220630235959.xlsx')
A5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL5_20220701000000_20220731235959.xlsx')
A6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL5_20220801000000_20220831235959.xlsx')
A7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL5_20220901000000_20220930235959.xlsx')
A8 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型A/CL5_20221001000000_20221031235959.xlsx')
CAR_A5 = pd.concat([A1, A2, A3, A4, A5, A6, A7, A8])

B1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL1_20220401000000_20220430235959.xlsx')
B2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL1_20220501000000_20220531235959.xlsx')
B3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL1_20220601000000_20220630235959.xlsx')
B4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL1_20220701000000_20220731235959.xlsx')
B5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL1_20220801000000_20220831235959.xlsx')
B6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL1_20220901000000_20220930235959.xlsx')
B7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL1_20221001000000_20221031235959.xlsx')
CAR_B1 = pd.concat([B1, B2, B3, B4, B5, B6, B7])

B1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL2_20220401000000_20220430235959.xlsx')
B2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL2_20220501000000_20220531235959.xlsx')
B3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL2_20220601000000_20220630235959.xlsx')
B4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL2_20220701000000_20220731235959.xlsx')
B5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL2_20220801000000_20220831235959.xlsx')
B6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL2_20220901000000_20220930235959.xlsx')
B7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL2_20221001000000_20221031235959.xlsx')
CAR_B2 = pd.concat([B1, B2, B3, B4, B5, B6, B7])

B1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL3_20220401000000_20220430235959.xlsx')
B2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL3_20220501000000_20220531235959.xlsx')
B3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL3_20220601000000_20220630235959.xlsx')
B4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL3_20220701000000_20220731235959.xlsx')
B5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL3_20220801000000_20220831235959.xlsx')
B6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL3_20220901000000_20220930235959.xlsx')
B7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL3_20221001000000_20221031235959.xlsx')
CAR_B3 = pd.concat([B1, B2, B3, B4, B5, B6, B7])

B1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL4_20220401000000_20220430235959.xlsx')
B2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL4_20220501000000_20220531235959.xlsx')
B3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL4_20220601000000_20220630235959.xlsx')
B4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL4_20220701000000_20220731235959.xlsx')
B5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL4_20220801000000_20220831235959.xlsx')
B6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL4_20220901000000_20220930235959.xlsx')
B7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL4_20221001000000_20221031235959.xlsx')
CAR_B4 = pd.concat([B1, B2, B3, B4, B5, B6, B7])

B1 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL5_20220401000000_20220430235959.xlsx')
B2 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL5_20220501000000_20220531235959.xlsx')
B3 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL5_20220601000000_20220630235959.xlsx')
B4 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL5_20220701000000_20220731235959.xlsx')
B5 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL5_20220801000000_20220831235959.xlsx')
B6 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL5_20220901000000_20220930235959.xlsx')
B7 = pd.read_excel(r'E:/PHEV DATA/EV-SZ/车型B/CL5_20221001000000_20221031235959.xlsx')
CAR_B5 = pd.concat([B1, B2, B3, B4, B5, B6, B7])



# %%
"""
    Descrtiption:

    深圳10辆车总数据量为：A型车：3411905，B型车：3156167 共 6568072
    采样时间从2022-4-1至2022-10-31 约7个月

"""
lengths1 = [len(df) for df in
           [CAR_A1,CAR_A2,CAR_A3,CAR_A4,CAR_A5]]

lengths2 = [len(df) for df in
           [CAR_B1,CAR_B2,CAR_B3,CAR_B4,CAR_B5]]

total_length = sum(lengths1+lengths2)

#%%
"""CNEV"""
V1 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE0K1A07972.csv')
V2 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE0K1B16707.csv')
V3 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE1K1B03139.csv')
V4 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE1K1B12505.csv')
V5 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE2K1B02940.csv')
V6 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE5K1A24931.csv')
V7 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE5K1B01524.csv')
V8 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE7K1B00195.csv')
V9 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE8K1A09808.csv')
V10 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE8K1B05048.csv')
V11 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PE9K1B01686.csv')
V12 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PEXK1B02846.csv')
V13 = pd.read_csv('E:\PHEV DATA\EV-CN/LFPHC7PEXK1B05312.csv')
# %%
"""
    Descrtiption:

    全国13辆车总数据量为：13186731

    采样时间从2021-7-30至2022-1-27 约半年

"""
lengths = [len(df) for df in
           [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13]]
total_length = sum(lengths)

#%%






"SHEV-200"









#%%
"""Model1"""
file_path1 = 'E:\PHEV DATA\EV-SH-200/vmodelseq=123114323/' # 以其中1种车型为例，没有直接写6个遍历
tmp1 = os.listdir(file_path1)

filenames1 = [int(name.split('.')[0]) for name in tmp1]
filenames1.sort()

Model1 = []
print('In',file_path1[2:-1],',',len(filenames1),'cars data will be loaded, please pay attention to the memory!')
for name in filenames1:
    Model1.append(pd.read_csv(file_path1 + str(name) +'.txt'))

"""Model2"""
file_path2 = 'E:\PHEV DATA\EV-SH-200/vmodelseq=123114716/' # 以其中1种车型为例，没有直接写6个遍历
tmp2 = os.listdir(file_path2)

filenames2 = [int(name.split('.')[0]) for name in tmp2]
filenames2.sort()

Model2 = []
print('In',file_path2[2:-1],',',len(filenames2),'cars data will be loaded, please pay attention to the memory!')
for name in filenames2:
    Model2.append(pd.read_csv(file_path2 + str(name) +'.txt'))

"""Model3"""
file_path3 = 'E:\PHEV DATA\EV-SH-200/vmodelseq=123114718/' # 以其中1种车型为例，没有直接写6个遍历
tmp3 = os.listdir(file_path3)

filenames3 = [int(name.split('.')[0]) for name in tmp3]
filenames3.sort()

Model3 = []
print('In',file_path3[2:-1],',',len(filenames3),'cars data will be loaded, please pay attention to the memory!')
for name in filenames3:
    Model3.append(pd.read_csv(file_path3 + str(name) +'.txt'))
#%%
"""
    部分数据Descrtiption:

    上海88辆车总数据量为：108，609，638
    车型1，39辆，数据量：27549737
    车型2，22辆，数据量：37093123
    车型3，27辆，数据量：43966778

    采样时间从2022-6-1至2023-6-30 约1年，时间不统一

"""

lengths1 = [len(df) for df in Model1]
total_length1 = sum(lengths1)
lengths2 = [len(df) for df in Model2]
total_length2 = sum(lengths2)
lengths3 = [len(df) for df in Model3]
total_length3 = sum(lengths3)
#%%

"""Model4-1"""
file_path4 = 'E:\PHEV DATA\EV-SH-200/vmodelseq=123114912/' # 以其中1种车型为例，没有直接写6个遍历
tmp4 = os.listdir(file_path4)

filenames4 = [int(name.split('.')[0]) for name in tmp4]
filenames4.sort()

Model4_1 = []
print('In',file_path4[2:-1],',',len(filenames4),'cars data will be loaded, please pay attention to the memory!')
for name in filenames4[:40]:
    Model4_1.append(pd.read_csv(file_path4 + str(name) +'.txt'))


#%%
"""
    部分数据Descrtiption:

    上海40辆车，车型4，总数据量为：65387739


    采样时间从2022-6-1至2023-6-30 约1年，时间不统一

"""

lengths1 = [len(df) for df in Model4_1]
total_length1 = sum(lengths1)

#%%
"""Model4-2"""
file_path4 = 'E:\PHEV DATA\EV-SH-200/vmodelseq=123114912/' # 以其中1种车型为例，没有直接写6个遍历
tmp4 = os.listdir(file_path4)

filenames4 = [int(name.split('.')[0]) for name in tmp4]
filenames4.sort()

Model4_2 = []
print('In',file_path4[2:-1],',',len(filenames4),'cars data will be loaded, please pay attention to the memory!')
for name in filenames4[40:]:
    Model4_2.append(pd.read_csv(file_path4 + str(name) +'.txt'))


#%%
"""
    部分数据Descrtiption:

    上海43车，车型4，总数据量为：73120682


    采样时间从2022-6-1至2023-6-30 约1年，时间不统一

"""

lengths1 = [len(df) for df in Model4_2]
total_length1 = sum(lengths1)


#%%
"""Model5"""
file_path5 = 'E:\PHEV DATA\EV-SH-200/vmodelseq=123115359/' # 以其中1种车型为例，没有直接写6个遍历
tmp5 = os.listdir(file_path5)

filenames5 = [int(name.split('.')[0]) for name in tmp5]
filenames5.sort()

Model5 = []
print('In',file_path5[2:-1],',',len(filenames5),'cars data will be loaded, please pay attention to the memory!')
for name in filenames5:
    Model5.append(pd.read_csv(file_path5 + str(name) +'.txt'))

"""Model6"""
file_path6 = 'E:\PHEV DATA\EV-SH-200/vmodelseq=123115361/' # 以其中1种车型为例，没有直接写6个遍历
tmp6 = os.listdir(file_path6)

filenames6 = [int(name.split('.')[0]) for name in tmp6]
filenames6.sort()

Model6 = []
print('In',file_path6[2:-1],',',len(filenames6),'cars data will be loaded, please pay attention to the memory!')
for name in filenames6:
    Model6.append(pd.read_csv(file_path6 + str(name) +'.txt'))
#%%
"""
    部分数据Descrtiption:

    上海23辆车，总数据量为：55,434,076
    车型5，1辆，数据量：999271
    车型6，22辆，数据量：54434805


    采样时间从2022-6-1至2023-6-30 约1年，时间不统一

"""

lengths1 = [len(df) for df in Model5]
total_length1 = sum(lengths1)
lengths2 = [len(df) for df in Model6]
total_length2 = sum(lengths2)
#%%

"""
    Descrtiption:

    上海199车，6种车型，总数据量为：302552135（3亿条）


    采样时间从2022-6-1至2023-6-30 约1年，时间不统一

"""

#%%
"SHEV-GPS"

V1 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=61.csv')
V2 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=92.csv')
V3 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=98.csv')
V4 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=465.csv')
V5 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=481.csv')
V6 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=123114461.csv')
V7 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=123114462.csv')
V8 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=123114606.csv')
V9 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=123114718.csv')
V10 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=123114737.csv')
V11 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=123114780.csv')
V12 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=123114800.csv')
V13 = pd.read_csv('E:\PHEV DATA\EV-SH-GPS/vmodelseq=123114882.csv')

lengths = [len(df) for df in [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13]]
total_length = sum(lengths)
unique_counts = [df['vin'].nunique() for df in [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13]]
#%%

"""
    Descrtiption:

    上海，车辆数159，13车型，，每种车型的车辆数为4, 1, 41, 1, 2, 31, 28, 14, 6, 6, 10, 10, 11带GPS总数据量为：10015461

    间隔30s
    
    采样时间从约6个月，时间非常不统一

"""

