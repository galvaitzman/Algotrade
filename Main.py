import pandas as pd
import pickle
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor


def save_best_model():
    """
    This function find the best model from the file best_reults.csv and save it into a pickle object.
    """
    best_results = pd.read_csv("datasets/best_results.csv")
    all_results = pd.read_csv("datasets/all_results.csv")
    df = pd.read_csv("datasets/aggregate_df.csv")
    best_result_index = best_results[['average MSE of 10 folds']].idxmin()
    all_results_index = all_results[['average MSE of 10 folds']].idxmin()
    model_name = list(best_results.loc[best_result_index, 'Model'])[0]
    param_value = float(list(all_results.loc[all_results_index, 'param value'])[0])
    features = list((best_results.iloc[best_result_index, 5:]).iloc[0])
    features = [x for x in features if str(x) != 'nan']
    model = create_model(model_name=model_name, param_value=param_value)
    X = df[features]
    y = df['SPY_return_Adj Close']
    model.fit(X,y)
    pickle.dump(model, open("best_model.pickle", "wb"))


def create_model(model_name, param_value):
    """
    This function return a new regression model of type model_name

    :param model_name: string

    :param param_value: int/float
    :return: regression model

    """
    if model_name == "LinearRegression":
        return linear_model.LinearRegression(normalize=param_value)
    if model_name == "linear_model.Ridge":
        return linear_model.Ridge(alpha=param_value)
    if model_name == "Lasso":
        return linear_model.Lasso(alpha=param_value)
    return RandomForestRegressor(n_estimators=param_value)


def load_best_model():
    model = pickle.load(open("best_model.pickle", "rb"))
    return model


# save_best_model()
load_best_model()