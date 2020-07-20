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
    all_dates_df = df.drop(columns=date_columns[1:])
    stock_columns.append('SPY_return_Adj Close')
    i = 0
    all_dates_df = all_dates_df.rename(columns={"XLY_return_Date": "Date"})

    all_dates_df.iloc[:, 1:] = np.nan

    while i <= (last_date.date() - first_date.date()).days:
        all_dates_df.at[i, 'Date'] = first_date + datetime.timedelta(days=i)
        all_dates_df.at[i, 'Is Beginning of a Month'] = (first_date + datetime.timedelta(days=i)).day < 15
        all_dates_df.at[i, 'Is Beginning of a Year'] = (first_date + datetime.timedelta(days=i)).month < 6
        i += 1

    for stock in stock_columns:
        stock_name = stock.split('_')[0]
        print(stock_name)
        date = [col for col in date_columns if stock_name in col]
        i = 0
        while i < len(df[date]):
            if type(df[date[0]][i]) is str :
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
    all_dates_df = pd.read_csv("all_dates_df.csv")
    aggregate_df = pd.DataFrame()

    tmp_date = first_date

    i = 0
    while tmp_date.date() < last_date.date():

        while i < 20:
            aggregate_df = aggregate_df.append(
                {'Date': str(tmp_date)[0:10] + " - " + str(tmp_date + datetime.timedelta(days=delta))[0:10],
                 'Stock Name': stock_columns[i]}
                , ignore_index=True)
            i += 1

        tmp_date = tmp_date + datetime.timedelta(days=delta + 1)
        i = 0

    df_dummies = pd.DataFrame(data=pd.get_dummies(aggregate_df['Stock Name']))
    aggregate_df = aggregate_df.join(df_dummies)
    print(aggregate_df.head())
    day_counter = 1
    for i in range(1, delta + 2):
        aggregate_df['Day ' + str(day_counter)] = np.nan
        day_counter += 1

    i = 0
    tmp_date = first_date
    j = 0

    while i < len(aggregate_df) and 0 <= (last_date.date() - tmp_date.date()).days:
        for day_counter in range(1, delta + 2):
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


def calculate_target():
    all_dates_df = pd.read_csv("all_dates_df.csv")
    aggregate_df = pd.read_csv("aggregate_df.csv")

    tmp_date = first_date

    # index over all_dates_df
    i = 0
    j = 0
    # index over aggregate_df
    index = 0

    arguments = []

    while i + delta < len(all_dates_df):

        while i + delta < len(all_dates_df) and j <= delta:
            arguments.append(all_dates_df.loc[i + delta, 'SPY_return_Adj Close'])
            j += 1
            i += 1
            
        avg = np.nanmean(arguments, axis=0)
        arguments = []
        j = 0
        while j < 20:
            aggregate_df.loc[index, 'SPY_return_Adj Close'] = avg
            index += 1
            j += 1
        j = 0

    aggregate_df.to_csv('aggregate_df.csv')


# def add_features():
#     aggregate_df = pd.read_csv("aggregate_df.csv")
#     all_dates_df = pd.read_csv("all_dates_df.csv")
#
#     j = 0
#     for index, row in aggregate_df.iterrows():
#         first_day = aggregate_df.loc[index, 'Date'][0:10]
#         i = 0
#         arguments = []
#         while i < delta:
#             arguments.append(all_dates_df.loc[j, "is end"])





create_all_dates_df()
#create_aggregate_df()
calculate_target()


