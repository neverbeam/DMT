import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    train_data = pd.read_csv('titanic_train.csv', delimiter = ',')
    train_data["Pclass"] = train_data["Pclass"].astype(str)
    print(train_data["Pclass"])
    print(np.unique(train_data["Pclass"]))

    plt.hist(train_data["Pclass"], bins=range(len(np.unique(train_data["Pclass"]))+1), align="left", rwidth=0.8)
    plt.xlabel("Pclass")
    plt.ylabel("count")
    plt.show()

    # for category in list(train_data.columns.values):
    #     plt.hist(train_data[category], bins=range(len(np.unique(train_data[category]))+1), align="left", rwidth=0.8)
    #     plt.xlabel(category)
    #     plt.ylabel("count")
    #     plt.show()