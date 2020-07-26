import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("all_results.csv")

fig, axs = plt.subplots(4, 4)
fig.suptitle('Parameters Graphs')
fig.set_figheight(15)
fig.set_figwidth(15)
fig.tight_layout()

i = 0
axs_counter_row = 0
axs_counter_col = 0


while i < len(df):
    model = df.loc[i, 'Model']
    param = df.loc[i, 'param']
    x = []
    y = []
    while True:
        x.append(df.loc[i, 'param value'])
        y.append(df.loc[i, 'average MSE of 10 folds'])
        i += 1
        if len(df) <= i or df.loc[i, 'Model'] != model:
            break

    axs[axs_counter_row][axs_counter_col].plot(x, y)
    axs[axs_counter_row][axs_counter_col].set(xlabel=param, ylabel='MSE')
    axs[axs_counter_row][axs_counter_col].set_title(model)
    axs_counter_col += 1

    if axs_counter_col == 4:
        axs_counter_row += 1
        axs_counter_col = 0


plt.show()