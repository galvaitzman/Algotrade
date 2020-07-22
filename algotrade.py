import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import datetime

df = pd.read_csv("Commodities_Return_S&P.csv")
date_columns = [col for col in df.columns if 'Date' in col]
stock_columns = [col for col in df.columns if col not in date_columns and col != 'SPY_return_Adj Close']
dates = df[date_columns[:]]

# first_date = datetime.datetime.strptime(dates.iloc[0].sort_values(ascending=True)[0], '%Y-%m-%d')
first_date = datetime.datetime.strptime('2010-02-02', '%Y-%m-%d')
last_date = datetime.datetime.strptime(dates.iloc[-1].sort_values(ascending=True)[0], '%Y-%m-%d')

delta = 5


def create_all_dates_df():
    """
    This function create a date frame which contains all the dates from first_date till end_date.
    Each row will contain the return_adj_close col of each stock from the original dataset
    and two more columns indicating if the date is from the beginning of a month and the beginning of the year.

    This function saves the new date frame to a file named - all_dates_df.csv
    """

    all_dates_df = df.drop(columns=date_columns[1:])
    stock_columns.append('SPY_return_Adj Close')
    i = 0
    all_dates_df = all_dates_df.rename(columns={"XLY_return_Date": "Date"})

    all_dates_df.iloc[:, 1:] = np.nan

    # create a row for each date starting from start date
    while i <= (last_date.date() - first_date.date()).days:
        all_dates_df.at[i, 'Date'] = first_date + datetime.timedelta(days=i)
        all_dates_df.at[i, 'Is Beginning of a Month'] = (first_date + datetime.timedelta(days=i)).day < 15
        all_dates_df.at[i, 'Is Beginning of a Year'] = (first_date + datetime.timedelta(days=i)).month < 6
        i += 1

    # copy the stock value from the orginal data frame to the new data frame
    for stock in stock_columns:
        stock_name = stock.split('_')[0]
        print(stock_name)
        date = [col for col in date_columns if stock_name in col]
        i = 0
        while i < len(df[date]):
            if type(df[date[0]][i]) is str:
                temp_date = datetime.datetime.strptime(df[date[0]][i], '%Y-%m-%d').date()
                if first_date.date() <= temp_date:
                    index = (datetime.datetime.strptime(df[date[0]][i], '%Y-%m-%d').date() - first_date.date()).days
                    all_dates_df[stock][index] = df[stock][i]
                i += 1
            else:
                break

    print(all_dates_df.head())
    all_dates_df.to_csv('all_dates_df.csv')


def create_aggregate_df():
    """
    This function create a data frame which be used to build our model.
    The new data frame will contain 20 rows (equal to the number of stocks in the original dataset) for each
    delta days interval.
    Each row will contain the stock value over the delta days

    """
    all_dates_df = pd.read_csv("all_dates_df.csv")
    aggregate_df = pd.DataFrame()

    tmp_date = first_date

    i = 0

    while tmp_date.date() < last_date.date():

        # add 20 lines for each interval
        while i < 20:
            aggregate_df = aggregate_df.append(
                {'Date': str(tmp_date)[0:10] + " - " + str(tmp_date + datetime.timedelta(days=delta - 1))[0:10],
                 'Stock Name': stock_columns[i]}
                , ignore_index=True)
            i += 1

        tmp_date = tmp_date + datetime.timedelta(days=delta)
        i = 0

    # create dummies for the stock names
    df_dummies = pd.DataFrame(data=pd.get_dummies(aggregate_df['Stock Name']))
    aggregate_df = aggregate_df.join(df_dummies)

    day_counter = 1

    # create delta columns for each day in the interval
    for i in range(1, delta + 1):
        aggregate_df['Day ' + str(day_counter)] = np.nan
        day_counter += 1

    i = 0
    tmp_date = first_date
    j = 0

    # add the relevant value of stock for each day
    while i < len(aggregate_df) and 0 <= (last_date.date() - tmp_date.date()).days:
        for day_counter in range(1, delta + 1):
            j = 0
            while j < 20:
                if 0 <= (last_date.date() - tmp_date.date()).days:
                    col = [col for col in aggregate_df.columns if aggregate_df.loc[j, col] == 1]
                    index = (tmp_date.date() - first_date.date()).days
                    aggregate_df['Day ' + str(day_counter)][i + j] = all_dates_df.loc[index, col]
                    j += 1
                else:
                    break
            tmp_date = tmp_date + datetime.timedelta(days=1)
        i += j

    aggregate_df.to_csv('aggregate_df.csv')

from sklearn import preprocessing
def fill_nan_values():
    # droping column
    if ('Unnamed: 0' in all_dates_df.columns):
        all_dates_df.drop(axis=1, columns=['Unnamed: 0'], inplace=True)

    # fill Nans
    columns = [col for col in all_dates_df.columns if col != 'SPY_return_Adj Close' and col != 'Date']
    # fill Nans between two values (the Nan value will be replaced with the value of upper and lower value in the same column)
    for i in columns:
        all_dates_df[i] = (all_dates_df[i].ffill() + all_dates_df[i].bfill()) / 2
    # fill Nans if there is no upper value (the Nan value will be replaced with the average value in the same column)
    all_dates_df = all_dates_df.fillna(all_dates_df.mean())

    # Normalizing all columns in range (-1,1)
    columns = all_dates_df.columns.copy()
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    min_max_scaler.fit_transform(all_dates_df.iloc[:, 1:])
    all_dates_df_temp = pd.DataFrame(min_max_scaler.transform(all_dates_df.iloc[:, 1:]))
    all_dates_df_temp.insert(loc=0, column='Date', value=all_dates_df['Date'])
    all_dates_df_temp.columns = columns
    all_dates_df = all_dates_df_temp


def add_dates_part(all_dates_df: pd.DataFrame, aggregate_df: pd.DataFrame):
    """
    This function add 2 new columns to the aggregate_df.
    The first new column will be named 'Is Beginning of a Month' and will have a value of 0 if
    most of the days in the interval belong to the beginning of the month, otherwise 1.
    The second new column will be named 'Is Beginning of a Year' and will have a value of 0 if
    most of the days in the interval belong to the month which belong to the beginning of the year, otherwise 1.

    :param all_dates_df: Data Frame
    :param aggregate_df: Date Frame
    :return: The updated data frame (aggregate_df)
    """

    # index over all_dates_df
    j = 0
    # index over aggregate_df
    index = 0
    while index < len(aggregate_df):

        counter = 1  # count every delta days
        month_arguments = []
        year_arguments = []

        while counter <= delta and j < len(all_dates_df):
            month_arguments.append(all_dates_df.loc[j, "Is Beginning of a Month"])
            year_arguments.append(all_dates_df.loc[j, "Is Beginning of a Year"])
            counter += 1
            j += 1

        month_avg = np.mean(month_arguments)
        year_avg = np.mean(year_arguments)

        k = index + 20

        while index < k:
            if month_avg < 0.5:  # majority of the days are in the second half of the month
                aggregate_df.loc[index, 'Is Beginning of a Month'] = 0
            else:
                aggregate_df.loc[index, 'Is Beginning of a Month'] = 1

            if year_avg < 0.5:  # the month is at the first half of the year
                aggregate_df.loc[index, 'Is Beginning of a Year'] = 0
            else:
                aggregate_df.loc[index, 'Is Beginning of a Year'] = 1
            index += 1

    return aggregate_df


def add_change_stock_between_two_following_days(aggregate_df: pd.DataFrame):
    """
    This function add delta - 1 columns which represent the change in stock for every two following days.
    :param aggregate_df: Data Frame

    :return: The updated data frame
    """
    for index, row in aggregate_df.iterrows():
        day_counter = 1

        while day_counter < delta:
            first_day_value = row['Day ' + str(day_counter)]
            second_day_value = row['Day ' + str(day_counter + 1)]
            difference = second_day_value - first_day_value
            aggregate_df.loc[index, 'Day ' + str(day_counter + 1) + '- Day ' + str(day_counter)] = difference
            day_counter += 1

    return aggregate_df


def add_features():
    """
    This function add new features to aggregate data frame.
    """
    all_dates_df = pd.read_csv("all_dates_df.csv")
    aggregate_df = pd.read_csv("aggregate_df.csv")
    # add 2 columns indicating if most of the days in the interval belongs to the beginning of the month and if the
    # interval month(s) belongs to the beginning og the year.
    aggregate_df = add_dates_part(all_dates_df=all_dates_df,
                                  aggregate_df=aggregate_df.iloc[:, 1:])

    # add the change in stocks for every two following days.
    aggregate_df = add_change_stock_between_two_following_days(aggregate_df=aggregate_df)

    aggregate_df.to_csv('aggregate_df.csv')


def calculate_target():
    """
    This function will calculate the target value and will write the values to a column named 'SPY_return_Adj Close'.
    The calculated target will be an average of the following delta days for each interval from the SPY values in the
    original data frame.

    """
    all_dates_df = pd.read_csv("all_dates_df.csv")
    aggregate_df = pd.read_csv("aggregate_df.csv")
    aggregate_df = aggregate_df.iloc[:, 1:]

    # index over all_dates_df
    i = 0
    j = 0
    # index over aggregate_df
    index = 0

    while i + delta < len(all_dates_df):

        arguments = []
        # collect the value of SPY return adj close over the next delta days
        while i + delta < len(all_dates_df) and j < delta:
            arguments.append(all_dates_df.loc[i + delta, 'SPY_return_Adj Close'])
            j += 1
            i += 1

        avg = np.nanmean(arguments, axis=0)

        j = 0
        # write the calculated avg in the current interval
        while j < 20:
            aggregate_df.loc[index, 'SPY_return_Adj Close'] = avg
            index += 1
            j += 1
        j = 0

    aggregate_df.to_csv('aggregate_df.csv')


def build_model(test_size):

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv("aggregate_df.csv")
    features = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 2- Day 1', 'Day 3- Day 2', 'Day 4- Day 3',
                'Day 5- Day 4', 'Is Beginning of a Month', 'Is Beginning of a Year']
    X = df[features]
    y = df['SPY_return_Adj Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # TODO add more relevant models
    models = [linear_model.LinearRegression(), linear_model.Ridge(alpha=.5)]
    models_train_accuracy = []
    MSE = []

    for model in models:
        # TODO not sure if we need to fit the model because we were requested to use k-cross validation
        model.fit(X=X_train, y=y_train)
        # 10 CROSS VALIDATION
        scores = cross_val_score(model, X, y, cv=10)
        accuracy = scores.mean()
        models_train_accuracy.append(accuracy)
        # TODO but we were requested to compare models using MSE so Im not sure what we're a supposed to do
        # predict x_test and calculate mse
        y_pred = model.predict(X=X_test)
        mse = mean_squared_error(y_test, y_pred)
        MSE.append(mse)

    # TODO choose the best model







# create_all_dates_df()
# create_aggregate_df()
# add_features()
# calculate_target()
# build_model(0.3)
