import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


df = pd.read_csv("/content/gdrive/My Drive/Algotrade/aggregate_df.csv")
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
columns = ['Model','average MSE of 10 folds','Time','n_estimators']
for i in range (len(all_features_for_prediction)):
  columns.append(str(i))
results = pd.DataFrame(columns=columns)
features = [features_to_use_1,features_to_use_2]
for i in features:
  current_run = []
  X = df[i]
  y = df['SPY_return_Adj Close']
  forest = RandomForestRegressor()
  n_estimators = [10,50,100,200]
  start_time = time.time()
  model_parameters = dict(n_estimators = n_estimators)
  grid = GridSearchCV(forest, model_parameters,refit=True, verbose = 10, n_jobs = -1,cv=10)
  best = grid.fit(X,y)
  current_run.append("RandomForestRegressor")
  current_run.append(str(-grid.cv_results_['mean_test_score'].max()))
  current_run.append(str(time.time() - start_time))
  current_run.append(grid.best_params_['n_estimators'])
  for j in i:
    current_run.append(j)
  while (len(current_run) < len(columns)):
    current_run.append(np.NaN)
  for counts in [current_run]:
    results.loc[len(results), :] = counts