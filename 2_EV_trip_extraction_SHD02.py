# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import pickle
warnings.filterwarnings("ignore")
#%%
def split_trips(V1):
    # 将时间戳转换为日期时间格式并按升序排序
    V1['collectiontime'] = pd.to_datetime(V1['collectiontime'] / 1000, unit='s')
    V1 = V1.sort_values(by='collectiontime')

    # 重置索引
    V1 = V1.reset_index(drop=True)

    # 找到从1开始的第一个索引
    start_index = V1['vehicledata_vehiclestatus'].eq(1).idxmax()

    # 保留从1开始的部分并删除之前的行
    V1 = V1.loc[start_index:]

    # 找到每次出行的起始和结束索引
    trip_start_indices = V1[(V1['vehicledata_vehiclestatus'] == 1) & (V1['vehicledata_vehiclestatus'].shift(1) != 1)].index.tolist()
    trip_end_indices = V1[(V1['vehicledata_vehiclestatus'] != 1) & (V1['vehicledata_vehiclestatus'].shift(-1) == 1)].index.tolist()

    # 创建多个子 DataFrame
    trip_dfs = [V1.loc[start:end] for start, end in zip(trip_start_indices, trip_end_indices)]

    # 过滤掉符合条件的行并拆分超过15分钟的部分
    new_trip_dfs = []
    for trip_df in trip_dfs:
        # 过滤掉 vehicledata_vehiclestatus 不等于 1 或 vehicledata_runmodel 不等于 1 的行
        trip_df = trip_df[(trip_df['vehicledata_vehiclestatus'] == 1) & (trip_df['vehicledata_runmodel'] == 1)]

        trip_df = trip_df[trip_df['vehicledata_chargestatus'] != 1]
        # 拆分超过15分钟的部分
        time_diff = trip_df['collectiontime'].diff()
        split_indices = trip_df[time_diff > pd.Timedelta(minutes=15)].index.tolist()

        if len(split_indices) > 0:
            split_indices = [trip_df.index[0]] + split_indices + [trip_df.index[-1]+1]
            for i in range(len(split_indices)-1):
                new_trip_dfs.append(trip_df.loc[split_indices[i]:split_indices[i+1]-1])
        else:
            new_trip_dfs.append(trip_df)

    return new_trip_dfs


def process_trip_data(new_trip_dfs):
    DF = pd.DataFrame(columns=['行程能耗','出行时间', '行程时间', '行程距离', '行程平均速度', '当前SOC',
                               '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差',
                               '绝缘电阻值'])

    for trip_df in new_trip_dfs:
        # (0) 行程能耗
        sum_current = trip_df['vehicledata_sumcurrent']
        sum_voltage = trip_df['vehicledata_sumvoltage']
        energy_consumption = np.abs(((sum_current * sum_voltage) * 10 / 1000 / 3600).sum())

        # (1) 出行时间
        trip_start_time = trip_df['collectiontime'].iloc[0]

        # (2) 行程时间
        trip_end_time = trip_df['collectiontime'].iloc[-1]
        trip_duration = (trip_end_time - trip_start_time).total_seconds() / 60

        # (3) 行程距离
        trip_distance = trip_df['vehicledata_summileage'].iloc[-1] - trip_df['vehicledata_summileage'].iloc[0]

        # (4) 行程平均速度
        trip_avg_speed = trip_df['vehicledata_speed'].mean()

        # (5) 当前 SOC
        current_soc = trip_df['vehicledata_soc'].iloc[0]

        # (6) 当前累积行驶里程
        current_mileage = trip_df['vehicledata_summileage'].iloc[0]

        # (7) 单体电池电压极差
        max_voltage = trip_df['extremevalue_maxbatterysinglevoltageval'].iloc[0]
        min_voltage = trip_df['extremevalue_minbatterysinglevoltageval'].iloc[0]
        voltage_range = max_voltage - min_voltage

        # (8) 单体电池温度极差
        max_temperature = trip_df['extremevalue_maxtmpval'].iloc[0]
        min_temperature = trip_df['extremevalue_mintmpval'].iloc[0]
        temperature_range = max_temperature - min_temperature

        # (9) 绝缘电阻值
        insulation_resistance = trip_df['vehicledata_insulationresistance'].iloc[0]

        feature = {
            '行程能耗': energy_consumption,
            '出行时间': trip_start_time,
            '行程时间': trip_duration,
            '行程距离': trip_distance,
            '行程平均速度': trip_avg_speed,
            '当前SOC': current_soc,
            '当前累积行驶里程': current_mileage,
            '单体电池电压极差': voltage_range,
            '单体电池温度极差': temperature_range,
            '绝缘电阻值': insulation_resistance,
        }
        DF = DF.append(feature, ignore_index=True)


    DF.loc[DF['行程距离'] == 0, '行程距离'] = (DF['行程时间'] * DF['行程平均速度'] / 60)

    # 时间筛选
    bool_index = (DF['行程时间'] >= 10) & (DF['行程时间'] < 1440)
    DF = DF[bool_index]

    #距离筛选
    bool_index = DF['行程距离'] >=1
    DF = DF[bool_index]

    #速度筛选
    bool_index = DF['行程平均速度'] >=1
    DF = DF[bool_index]

    #SOC筛选
    bool_index = (DF['当前SOC'] >=5) & (DF['当前SOC'] <= 100)
    DF = DF[bool_index]

    # 能耗筛选
    bool_index = (DF['行程能耗'] >=0.1) & (DF['行程能耗'] <= 200)
    DF = DF[bool_index]


    # 计算 Z-score
    DF['单体电池电压极差'] = DF['单体电池电压极差'].astype(float)
    DF['单体电池温度极差'] = DF['单体电池温度极差'].astype(float)
    DF['绝缘电阻值'] = DF['绝缘电阻值'].astype(float)
    DF['单体电池电压极差_zscore'] = stats.zscore(DF['单体电池电压极差'])
    DF['单体电池温度极差_zscore'] = stats.zscore(DF['单体电池温度极差'])
    DF['绝缘电阻值_zscore'] = stats.zscore(DF['绝缘电阻值'])

    outliers1 = DF[(DF['单体电池电压极差_zscore'] > 3) | (DF['单体电池电压极差_zscore'] < -3)]
    outliers2 = DF[(DF['单体电池温度极差_zscore'] > 3) | (DF['单体电池温度极差_zscore'] < -3)]
    outliers3 = DF[(DF['绝缘电阻值_zscore'] > 3) | (DF['绝缘电阻值_zscore'] < -3)]

    DF.loc[outliers1.index, '单体电池电压极差'] = DF['单体电池电压极差'].mean()
    DF.loc[outliers2.index, '单体电池温度极差'] = DF['单体电池温度极差'].mean()
    DF.loc[outliers3.index, '绝缘电阻值'] = DF['绝缘电阻值'].mean()

    DF.drop(['单体电池电压极差_zscore', '单体电池温度极差_zscore', '绝缘电阻值_zscore'], axis=1, inplace=True)

    return DF


def merge_weather_data(DF, SH_Weather):
    # 填充天气数据的NaN值为均值
    SH_Weather_filled = SH_Weather.fillna(SH_Weather.mean())

    # 创建一个空的DataFrame来存储匹配后的数据
    combined_data = pd.DataFrame()

    # 对于EV行程数据的每一行
    for index, row in DF.iterrows():
        # 找到最接近的天气时间点
        closest_weather_time = SH_Weather_filled.iloc[
            (SH_Weather_filled.iloc[:, 0] - row['出行时间']).abs().argsort()[:1]]

        # 将EV行程数据与最接近的天气数据合并
        combined_row = pd.concat([pd.DataFrame(row).transpose().reset_index(drop=True),
                                  closest_weather_time.iloc[:, 1:].reset_index(drop=True)], axis=1)

        # 将合并后的数据添加到combined_data中
        combined_data = combined_data.append(combined_row, ignore_index=True)

    # 添加出行季节列
    combined_data['出行季节'] = combined_data['出行时间'].apply(lambda x: 'spring' if x.month in [3, 4, 5] else (
        'summer' if x.month in [6, 7, 8] else ('autumn' if x.month in [9, 10, 11] else 'winter')))

    # 添加出行日期列
    combined_data['出行日期'] = combined_data['出行时间'].apply(lambda x: x.strftime('%A'))

    # 添加出行时段列
    combined_data['出行时段'] = combined_data['出行时间'].apply(lambda x: 'morning peak' if 7 <= x.hour < 10 else (
        'night peak' if 15 <= x.hour < 18 else (
            'nighttime' if 22 <= x.hour <= 24 or 0 <= x.hour < 7 else 'other time')))

    return combined_data

#%%
SHD02Y22M1=[]
# 循环处理V1到V20的DataFrame并得到相应的combined_data
for i in range(39):
    # 生成变量名

    # 处理数据
    new_trip_dfs = split_trips(Model1[i])
    new_trip_dfs = [df for df in new_trip_dfs if len(df) >= 30]
    DF = process_trip_data(new_trip_dfs)
    SH_Weather = pd.read_excel('11-pretrained_incremental_learning/SH W.xls')
    SH_Weather.iloc[:, 0] = pd.to_datetime(SH_Weather.iloc[:, 0], format="%d.%m.%Y %H:%M")
    combined_data = merge_weather_data(DF, SH_Weather)
    combined_data['电池能量'] = 35
    combined_data['车辆类型'] = 'Sedan'
    combined_data['整备质量'] = 1015
    combined_data['地点'] = 'SH'

    combined_data['VIN'] = f'SHD02Y22V{i}'

    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']] = \
    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']].astype(
        float)
    combined_data = combined_data[combined_data['行程能耗'] <= combined_data['电池能量']]

    SHD02Y22M1.append(combined_data)


combined_df = pd.concat(SHD02Y22M1, ignore_index=True)
A=combined_df .describe()
#%%



with open('11-pretrained_incremental_learning/data/SHD02Y22M1.pkl', 'wb') as f:
    pickle.dump(SHD02Y22M1, f)



#%%
SHD02Y22M2=[]

for i in range(22):

    # 处理数据
    new_trip_dfs = split_trips(Model2[i])
    new_trip_dfs = [df for df in new_trip_dfs if len(df) >= 30]
    DF = process_trip_data(new_trip_dfs)
    SH_Weather = pd.read_excel('11-pretrained_incremental_learning/SH W.xls')
    SH_Weather.iloc[:, 0] = pd.to_datetime(SH_Weather.iloc[:, 0], format="%d.%m.%Y %H:%M")
    combined_data = merge_weather_data(DF, SH_Weather)
    combined_data['电池能量'] = 50.8
    combined_data['车辆类型'] = 'Sedan'
    combined_data['整备质量'] = 1550
    combined_data['地点'] = 'SH'

    combined_data['VIN'] = f'SHD02Y22V{i}'

    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']] = \
    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']].astype(
        float)

    combined_data = combined_data[combined_data['行程能耗'] <= combined_data['电池能量']]

    SHD02Y22M2.append(combined_data)


combined_df = pd.concat(SHD02Y22M2, ignore_index=True)
A=combined_df .describe()
#%%

with open('11-pretrained_incremental_learning/data/SHD02Y22M2.pkl', 'wb') as f:
    pickle.dump(SHD02Y22M2, f)

#%%
SHD02Y22M3=[]

for i in range(27):

    # 处理数据
    new_trip_dfs = split_trips(Model3[i])
    new_trip_dfs = [df for df in new_trip_dfs if len(df) >= 30]
    DF = process_trip_data(new_trip_dfs)
    SH_Weather = pd.read_excel('11-pretrained_incremental_learning/SH W.xls')
    SH_Weather.iloc[:, 0] = pd.to_datetime(SH_Weather.iloc[:, 0], format="%d.%m.%Y %H:%M")
    combined_data = merge_weather_data(DF, SH_Weather)
    combined_data['电池能量'] = 52.5
    combined_data['车辆类型'] = 'Sedan'
    combined_data['整备质量'] = 1560
    combined_data['地点'] = 'SH'

    combined_data['VIN'] = f'SHD02Y22V{i}'

    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']] = \
    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']].astype(
        float)

    combined_data = combined_data[combined_data['行程能耗'] <= combined_data['电池能量']]

    SHD02Y22M3.append(combined_data)


combined_df = pd.concat(SHD02Y22M3, ignore_index=True)
A=combined_df .describe()
#%%

with open('11-pretrained_incremental_learning/data/SHD02Y22M3.pkl', 'wb') as f:
    pickle.dump(SHD02Y22M3, f)

#%%
SHD02Y22M4_1=[]

for i in range(40):

    # 处理数据
    new_trip_dfs = split_trips(Model4_1[i])
    new_trip_dfs = [df for df in new_trip_dfs if len(df) >= 30]
    DF = process_trip_data(new_trip_dfs)
    SH_Weather = pd.read_excel('11-pretrained_incremental_learning/SH W.xls')
    SH_Weather.iloc[:, 0] = pd.to_datetime(SH_Weather.iloc[:, 0], format="%d.%m.%Y %H:%M")
    combined_data = merge_weather_data(DF, SH_Weather)
    combined_data['电池能量'] = 50.8
    combined_data['车辆类型'] = 'Sedan'
    combined_data['整备质量'] = 1640
    combined_data['地点'] = 'SH'

    combined_data['VIN'] = f'SHD02Y22V{i}'

    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']] = \
    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']].astype(
        float)

    combined_data = combined_data[combined_data['行程能耗'] <= combined_data['电池能量']]

    SHD02Y22M4_1.append(combined_data)


combined_df = pd.concat(SHD02Y22M4_1, ignore_index=True)
A=combined_df .describe()
#%%
SHD02Y22M4_2=[]

for i in range(43):

    # 处理数据
    new_trip_dfs = split_trips(Model4_2[i])
    new_trip_dfs = [df for df in new_trip_dfs if len(df) >= 30]
    DF = process_trip_data(new_trip_dfs)
    SH_Weather = pd.read_excel('11-pretrained_incremental_learning/SH W.xls')
    SH_Weather.iloc[:, 0] = pd.to_datetime(SH_Weather.iloc[:, 0], format="%d.%m.%Y %H:%M")
    combined_data = merge_weather_data(DF, SH_Weather)
    combined_data['电池能量'] = 50.8
    combined_data['车辆类型'] = 'Sedan'
    combined_data['整备质量'] = 1640
    combined_data['地点'] = 'SH'

    combined_data['VIN'] = f'SHD02Y22V{40+i}'

    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']] = \
    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']].astype(
        float)

    combined_data = combined_data[combined_data['行程能耗'] <= combined_data['电池能量']]

    SHD02Y22M4_2.append(combined_data)


combined_df = pd.concat(SHD02Y22M4_2, ignore_index=True)
A=combined_df .describe()
#%%
SHD02Y22M4=SHD02Y22M4_1+SHD02Y22M4_2
#%%



with open('11-pretrained_incremental_learning/data/SHD02Y22M4.pkl', 'wb') as f:
    pickle.dump(SHD02Y22M4, f)


#%%
SHD02Y22M5=[]

for i in range(1):

    # 处理数据
    new_trip_dfs = split_trips(Model5[i])
    new_trip_dfs = [df for df in new_trip_dfs if len(df) >= 30]
    DF = process_trip_data(new_trip_dfs)
    SH_Weather = pd.read_excel('11-pretrained_incremental_learning/SH W.xls')
    SH_Weather.iloc[:, 0] = pd.to_datetime(SH_Weather.iloc[:, 0], format="%d.%m.%Y %H:%M")
    combined_data = merge_weather_data(DF, SH_Weather)
    combined_data['电池能量'] = 50.3
    combined_data['车辆类型'] = 'Sedan'
    combined_data['整备质量'] = 1880
    combined_data['地点'] = 'SH'

    combined_data['VIN'] = f'SHD02Y22V{i}'

    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']] = \
    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']].astype(
        float)

    combined_data = combined_data[combined_data['行程能耗'] <= combined_data['电池能量']]

    SHD02Y22M5.append(combined_data)


combined_df = pd.concat(SHD02Y22M5, ignore_index=True)
A=combined_df .describe()
#%%

with open('11-pretrained_incremental_learning/data/SHD02Y22M5.pkl', 'wb') as f:
    pickle.dump(SHD02Y22M5, f)

#%%
SHD02Y22M6=[]

for i in range(27):

    # 处理数据
    new_trip_dfs = split_trips(Model6[i])
    new_trip_dfs = [df for df in new_trip_dfs if len(df) >= 30]
    DF = process_trip_data(new_trip_dfs)
    SH_Weather = pd.read_excel('11-pretrained_incremental_learning/SH W.xls')
    SH_Weather.iloc[:, 0] = pd.to_datetime(SH_Weather.iloc[:, 0], format="%d.%m.%Y %H:%M")
    combined_data = merge_weather_data(DF, SH_Weather)
    combined_data['电池能量'] = 61.1
    combined_data['车辆类型'] = 'Sedan'
    combined_data['整备质量'] = 1560
    combined_data['地点'] = 'SH'

    combined_data['VIN'] = f'SHD02Y22V{i}'

    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']] = \
    combined_data[["行程能耗", "行程时间", '行程距离', '行程平均速度', '当前SOC'
        , '当前累积行驶里程', '单体电池电压极差', '单体电池温度极差', '绝缘电阻值', '电池能量', '整备质量']].astype(
        float)

    combined_data = combined_data[combined_data['行程能耗'] <= combined_data['电池能量']]

    SHD02Y22M6.append(combined_data)


combined_df = pd.concat(SHD02Y22M6, ignore_index=True)
A=combined_df .describe()
#%%

with open('11-pretrained_incremental_learning/data/SHD02Y22M6.pkl', 'wb') as f:
    pickle.dump(SHD02Y22M6, f)

