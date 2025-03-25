#%%
import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from My_utils.evaluation_scheme import evaluation
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")
#%%
# Load data
with open('/home/ps/haichao/11-pretrained_incremental_learning/data/Test_vehicle.pkl', 'rb') as file:
    Test_vehicle = pickle.load(file)

def labeling(Array):
    """Label encoding"""
    season_mapping = {'spring': 0, 'summer': 0.333, 'autumn': 0.667, 'winter': 1}
    Array['出行季节'] = Array['出行季节'].map(season_mapping)

    day_mapping = {'Monday': 0, 'Tuesday': 0.167, 'Wednesday': 0.333, 'Thursday': 0.5,
                   'Friday': 0.667, 'Saturday': 0.883, 'Sunday': 1}
    Array['出行日期'] = Array['出行日期'].map(day_mapping)

    period_mapping = {'morning peak': 0, 'night peak': 0.333, 'other time': 0.667, "nighttime": 1}
    Array['出行时段'] = Array['出行时段'].map(period_mapping)

    vehicle_mapping = {'Sedan': 0, 'SUV': 0.333, 'Sedan PHEV': 0.667, 'SUV PHEV': 1}
    Array['车辆类型'] = Array['车辆类型'].map(vehicle_mapping)

    Array['整备质量'] = Array['整备质量'] / 1880
    Array['电池能量'] = Array['电池能量'] / 61.1
    Array['当前累积行驶里程'] = Array['当前累积行驶里程'] / 500000

    Array.loc[Array['VV'].apply(lambda x: isinstance(x, str)), 'VV'] = 0.1

    columns_to_drop = ["出行时间", "地点", "VIN"]
    Array = Array.drop(columns=columns_to_drop).astype(float)
    Array = Array.fillna(0)

    return Array

SUM_M = []
P_all = []

for p in [32]:
    M = []
    test_all_simu = []
    test_all_real = []
    for arr in Test_vehicle:
        # arr = Test_vehicle[5]
        Array = labeling(arr)
        np.random.seed(42)

        # Shuffle and prepare the data
        Array = Array.reset_index(drop=True)
        x = Array.values[:, 1:]
        y = Array.values[:, 0]

        train_size = p
        test_size = int(len(x) * 0.3)

        trainX = Array.iloc[:train_size, 1:].values
        trainY = Array.iloc[:train_size, 0].values
        testX = Array.iloc[-test_size:, 1:].values
        testY = Array.iloc[-test_size:, 0].values

        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [10, 50, 100],  # Number of boosting rounds
            'learning_rate': [0.01, 0.1, 1],  # Learning rate shrinks the contribution of each regressor
            'base_estimator__max_depth': [2, 4, 6]  # Maximum depth of each tree in AdaBoost
        }

        # Initialize the base estimator
        base_estimator = DecisionTreeRegressor(random_state=42)

        # Define the AdaBoost regressor with the base estimator
        ada_regressor = AdaBoostRegressor(base_estimator=base_estimator, random_state=42)

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=ada_regressor,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # Negative MSE as a scoring metric for regression
            cv=2,  # Cross-validation folds
            n_jobs=-1  # Use all available cores
        )

        # Fit the model using grid search
        grid_search.fit(trainX, trainY)

        # Get the best model from grid search
        best_ada_regressor = grid_search.best_estimator_
        # print(f"Best parameters: {grid_search.best_params_}")

        # Predict using the best model
        EC_Pred1 = best_ada_regressor.predict(testX)

        # 定义随机森林的参数网格
        param_grid = {
            'n_estimators': [5, 10, 20],
            'max_depth': [None, 1, 3, 5],
        }

        # 初始化随机森林回归器
        rf_regressor = RandomForestRegressor(random_state=42)

        # 在内部循环中进行网格搜索
        grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=2)
        grid_search.fit(trainX, trainY)

        # 使用最佳参数的模型进行预测
        best_rf_regressor = grid_search.best_estimator_
        EC_Pred2 = best_rf_regressor.predict(testX)

        # Define hyperparameter grid for MLP
        param_grid = {
            'hidden_layer_sizes': [(10,10), (20,20), (30, 30)],  # Different configurations of hidden layers
            'learning_rate_init': [0.0001,0.001, 0.01, 0.1],  # Learning rate
        }

        # Initialize MLP regressor
        mlp_regressor = MLPRegressor(random_state=42)

        # Create GridSearchCV object
        grid_search = GridSearchCV(estimator=mlp_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=2)

        # Perform grid search
        grid_search.fit(trainX, trainY)

        # Get the best model
        best_mlp_regressor = grid_search.best_estimator_

        # Predict using the best model
        EC_Pred3 = best_mlp_regressor.predict(testX)

        EC_Pred=(EC_Pred1+EC_Pred2+EC_Pred3)/3
        EC_True = testY

        # Evaluate the predictions
        Metric1 = np.array(evaluation(EC_True, EC_Pred))
        print( Metric1)
        M.append(Metric1)

        test_all_simu.append(EC_Pred.reshape(-1, 1))
        test_all_real.append(EC_True.reshape(-1, 1))
        # break

    M2_test_all = np.array(evaluation(np.vstack(test_all_real), np.vstack(test_all_simu)))
    P_all.append(M2_test_all)

    METRIC = np.vstack(M)
    SUM_M.append(METRIC)

pd.DataFrame(SUM_M[-1])
#%%
data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/adaboost_data.pkl', 'wb') as f:
    pickle.dump(data, f)

#%% for scatter
indices_to_remove = [0, 15, 16, 17, 18]

# Iterate over the indices in reverse order to avoid index shifting issues while deleting
for index in sorted(indices_to_remove, reverse=True):
    del test_all_simu[index]
    del test_all_real[index]

EC_Pred = np.concatenate(test_all_simu, axis=0)
EC_True = np.concatenate(test_all_real, axis=0)

data = {'EC_True': EC_True.reshape(-1, 1), 'EC_Pred': EC_Pred.reshape(-1, 1)}

# 保存为 pickle 文件
with open('/home/ps/haichao/11-pretrained_incremental_learning/data_plot/adaboost_scatter.pkl', 'wb') as f:
    pickle.dump(data, f)