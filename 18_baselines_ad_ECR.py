# -*- coding: utf-8 -*-
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
from My_utils.evaluation_scheme import evaluation
import pandas as pd
import time
#%%

"""manufacturer provided ECR"""

#%%
"""loacl"""
with open('11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)
"""修正纯电里程"""
# for df_index in range(15, 19):
#     Test_vehicle[df_index]['行程距离'] *= 0.2


M=[]
Advtised_ECR=[16,16,13.3,13.3,11.6,
              11.6,12.6,12.6,13,13,
              13.3,13.3,13.3,13.3,13.3,
              22.0,14.1,14.1,20.0,13.5]
for i,arr in enumerate(Test_vehicle):

    testX = arr["行程距离"]
    testY = arr["行程能耗"]

    # 记录起始时间
    start_time = time.time()
    EC_True = testY
    EC_Pred = testX.values/100*Advtised_ECR[i]
    M.append(np.array(evaluation(EC_True, EC_Pred)))
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    execution_time = (end_time - start_time) / len(testX)
    print(f"代码运行时间: {execution_time} 秒")

METRIC=np.vstack(M)
#%%
"""charge car"""

"""ECR unknown"""
with open("11-pretrained_incremental_learning/data/ChargeCar.pkl", "rb") as f:
    ChargeCar = pickle.load(f)

#%%
"""BHD"""

with open("11-pretrained_incremental_learning/data/BHD.pkl", "rb") as f:
    BHD = pickle.load(f)

# 记录起始时间
start_time = time.time()
Advtised_ECR=16.3
testX = BHD["Distance [km]"]
testY = BHD["行程能耗"]

EC_True = testY

EC_Pred = testX.values / 100 * Advtised_ECR
np.array(evaluation(EC_True, EC_Pred))
# 记录结束时间
end_time = time.time()

# 计算运行时间
execution_time = (end_time - start_time)/len(testX)
print(f"代码运行时间: {execution_time} 秒")
#%%
"""SpritMonitor"""

with open("11-pretrained_incremental_learning/data/SpritMonitor.pkl", "rb") as f:
    SpritMonitor = pickle.load(f)


M=[]
Advtised_ECR=[10,13.6]
for i,arr in enumerate(SpritMonitor):

    testX = arr["trip_distance(km)"]
    testY = arr["quantity(kWh)"]

    # 记录起始时间
    start_time = time.time()
    EC_True = testY
    EC_Pred = testX.values/100*Advtised_ECR[i]
    M.append(np.array(evaluation(EC_True, EC_Pred)))
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    execution_time = (end_time - start_time) / len(testX)
    print(f"代码运行时间: {execution_time} 秒")

METRIC=np.vstack(M)


#%%
"""VED"""

with open("11-pretrained_incremental_learning/data/VED.pkl", "rb") as f:
    VED = pickle.load(f)

VED=pd.concat(VED, ignore_index=True)
Advtised_ECR=18.0
testX = VED["行程距离"]
testY = VED["行程能耗"]

# 记录起始时间
start_time = time.time()

EC_True = testY
EC_Pred = testX.values / 100 * Advtised_ECR
np.array(evaluation(EC_True, EC_Pred))

# 记录结束时间
end_time = time.time()

# 计算运行时间
execution_time = (end_time - start_time)/len(testX)
print(f"代码运行时间: {execution_time} 秒")