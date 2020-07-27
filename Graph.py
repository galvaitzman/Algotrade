import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("datasets/all_results.csv")

i = 0


features = "feature_to_use_"
string_number = "1"
number = 1

while i < len(df):
    model = df.loc[i, 'Model']
    param = df.loc[i, 'param']
    x = []
    y = []
    # collect all the rows with the same model name in the col 'Model'
    while True:
        x.append(df.loc[i, 'param value'])
        y.append(df.loc[i, 'average MSE of 10 folds'])
        i += 1
        if len(df) <= i or df.loc[i, 'Model'] != model:
            break

    # plot the data
    plt.plot(x, y)
    plt.xlabel(param)
    plt.ylabel('MSE')
    plt.title(model + " with " + features + string_number)
    plt.show()

    if i == 12 or i == 24 or i == 36:
        number += 1
        string_number = str(number)
