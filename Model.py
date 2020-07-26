import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

df = pd.read_csv("aggregate_df.csv")

all_features = ['Date', 'Stock Name',
                'BIL_return_Adj Close', 'EEM_return_Adj Close', 'GLD_return_Adj Close',
                'GSG_return_Adj Close', 'IEI_return_Adj Close', 'LQD_return_Adj Close',
                'SPHY_return_Adj Close', 'UUP_return_Adj Close', 'VNQ_return_Adj Close',
                'VWO_return_Adj Close', 'XLB_return_Adj Close', 'XLE_return_Adj Close',
                'XLF_return_Adj Close', 'XLI_return_Adj Close', 'XLK_return_Adj Close',
                'XLP_return_Adj Close', 'XLU_return_Adj Close', 'XLV_return_Adj Close',
                'XLY_return_Adj Close', 'XTL_return_Adj Close', 'Day 1', 'Day 2',
                'Day 3', 'Day 4', 'Day 5', 'Is Beginning of a Month',
                'Is Beginning of a Year', 'Day 2- Day 1', 'Day 3- Day 2',
                'Day 4- Day 3', 'Day 5- Day 4', 'SPY_return_Adj Close',
                'Positive Trends', 'Negative Trends', 'Is Positive Trend']
all_features_for_prediction = [
    'BIL_return_Adj Close', 'EEM_return_Adj Close', 'GLD_return_Adj Close',
    'GSG_return_Adj Close', 'IEI_return_Adj Close', 'LQD_return_Adj Close',
    'SPHY_return_Adj Close', 'UUP_return_Adj Close', 'VNQ_return_Adj Close',
    'VWO_return_Adj Close', 'XLB_return_Adj Close', 'XLE_return_Adj Close',
    'XLF_return_Adj Close', 'XLI_return_Adj Close', 'XLK_return_Adj Close',
    'XLP_return_Adj Close', 'XLU_return_Adj Close', 'XLV_return_Adj Close',
    'XLY_return_Adj Close', 'XTL_return_Adj Close', 'Day 1', 'Day 2',
    'Day 3', 'Day 4', 'Day 5', 'Is Beginning of a Month',
    'Is Beginning of a Year', 'Day 2- Day 1', 'Day 3- Day 2',
    'Day 4- Day 3', 'Day 5- Day 4', 'SPY_return_Adj Close',
    'Positive Trends', 'Negative Trends', 'Is Positive Trend']
features_to_use_1 = [
    'BIL_return_Adj Close', 'EEM_return_Adj Close', 'GLD_return_Adj Close',
    'GSG_return_Adj Close', 'IEI_return_Adj Close', 'LQD_return_Adj Close',
    'SPHY_return_Adj Close', 'UUP_return_Adj Close', 'VNQ_return_Adj Close',
    'VWO_return_Adj Close', 'XLB_return_Adj Close', 'XLE_return_Adj Close',
    'XLF_return_Adj Close', 'XLI_return_Adj Close', 'XLK_return_Adj Close',
    'XLP_return_Adj Close', 'XLU_return_Adj Close', 'XLV_return_Adj Close',
    'XLY_return_Adj Close', 'XTL_return_Adj Close', 'Day 1', 'Day 2',
    'Day 3', 'Day 4', 'Day 5']
features_to_use_2 = [
    'BIL_return_Adj Close', 'EEM_return_Adj Close', 'GLD_return_Adj Close',
    'GSG_return_Adj Close', 'IEI_return_Adj Close', 'LQD_return_Adj Close',
    'SPHY_return_Adj Close', 'UUP_return_Adj Close', 'VNQ_return_Adj Close',
    'VWO_return_Adj Close', 'XLB_return_Adj Close', 'XLE_return_Adj Close',
    'XLF_return_Adj Close', 'XLI_return_Adj Close', 'XLK_return_Adj Close',
    'XLP_return_Adj Close', 'XLU_return_Adj Close', 'XLV_return_Adj Close',
    'XLY_return_Adj Close', 'XTL_return_Adj Close', 'Day 2- Day 1', 'Day 3- Day 2',
    'Day 4- Day 3', 'Day 5- Day 4']
features_to_use_3 = [
    'BIL_return_Adj Close', 'EEM_return_Adj Close', 'GLD_return_Adj Close',
    'GSG_return_Adj Close', 'IEI_return_Adj Close', 'LQD_return_Adj Close',
    'SPHY_return_Adj Close', 'UUP_return_Adj Close', 'VNQ_return_Adj Close',
    'VWO_return_Adj Close', 'XLB_return_Adj Close', 'XLE_return_Adj Close',
    'XLF_return_Adj Close', 'XLI_return_Adj Close', 'XLK_return_Adj Close',
    'XLP_return_Adj Close', 'XLU_return_Adj Close', 'XLV_return_Adj Close',
    'XLY_return_Adj Close', 'XTL_return_Adj Close', 'Day 2- Day 1', 'Day 3- Day 2',
    'Day 4- Day 3', 'Day 5- Day 4', 'Positive Trends', 'Negative Trends', 'Is Positive Trend']
features_to_use_4 = [
    'BIL_return_Adj Close', 'EEM_return_Adj Close', 'GLD_return_Adj Close',
    'GSG_return_Adj Close', 'IEI_return_Adj Close', 'LQD_return_Adj Close',
    'SPHY_return_Adj Close', 'UUP_return_Adj Close', 'VNQ_return_Adj Close',
    'VWO_return_Adj Close', 'XLB_return_Adj Close', 'XLE_return_Adj Close',
    'XLF_return_Adj Close', 'XLI_return_Adj Close', 'XLK_return_Adj Close',
    'XLP_return_Adj Close', 'XLU_return_Adj Close', 'XLV_return_Adj Close',
    'XLY_return_Adj Close', 'XTL_return_Adj Close', 'Day 2- Day 1', 'Day 3- Day 2',
    'Day 4- Day 3', 'Day 5- Day 4', 'Is Beginning of a Month',
    'Is Beginning of a Year', 'Positive Trends', 'Negative Trends', 'Is Positive Trend']

columns = ['Model', 'average MSE of 10 folds', 'Time', 'param']
columns_for_all_results = columns.copy()
columns_for_all_results.append('param value')

for i in range(len(all_features_for_prediction)):
    columns.append(str(i))
    columns_for_all_results.append(str(i))

best_results = pd.DataFrame(columns=columns)
all_results = pd.DataFrame(columns=columns_for_all_results)

features = [features_to_use_1, features_to_use_2, features_to_use_3, features_to_use_4]

models = {"LinearRegression": [linear_model.LinearRegression(), dict(normalize=[False, True])],
          "linear_model.Ridge": [linear_model.Ridge(), dict(alpha=[0.01, 0.05, 0.1])],
          "Lasso": [linear_model.Lasso(), dict(alpha=[0.01, 0.05, 0.1])],
          "RandomForestRegressor": [RandomForestRegressor(), dict(n_estimators=[10, 50, 100, 200])]}

for i in features:
    for key, value in models.items():
        current_run_for_best_result = []
        current_run_for_all_results = []
        X = df[i]
        y = df['SPY_return_Adj Close']
        start_time = time.time()
        model = value[0]
        model_parameters = value[1]
        grid = GridSearchCV(model, model_parameters, refit=True, verbose=10, n_jobs=-1, cv=10)
        best = grid.fit(X, y)
        # collect data of the best model with the best params
        current_run_for_best_result.append(key)
        current_run_for_best_result.append(str(-grid.cv_results_['mean_test_score'].max()))
        current_run_for_best_result.append(str(time.time() - start_time))
        current_run_for_best_result.append(grid.best_params_.keys())

        # collect data for every param
        grid.cv_results_['mean_test_score'] = grid.cv_results_['mean_test_score'] * (-1)

        for param, param_values in value[1].items():
            counter = 0
            for param_value in param_values:
                current_run_for_all_results.append(
                    [key, grid.cv_results_['mean_test_score'][counter], str(time.time() - start_time), param,
                     param_value])
                for j in i:
                    current_run_for_all_results[len(current_run_for_all_results) - 1].append(j)
                while len(current_run_for_all_results[len(current_run_for_all_results) - 1]) < len(
                        columns_for_all_results):
                    current_run_for_all_results[len(current_run_for_all_results) - 1].append(np.NaN)
                counter += 1

        for j in i:
            current_run_for_best_result.append(j)
        while len(current_run_for_best_result) < len(columns):
            current_run_for_best_result.append(np.NaN)
        for counts in [current_run_for_best_result]:
            best_results.loc[len(best_results), :] = counts
        for data in current_run_for_all_results:
            all_results.loc[len(all_results), :] = data

best_results.to_csv("best_results.csv")
all_results.to_csv("all_results.csv")
