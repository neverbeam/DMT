import pandas as pd
import matplotlib.pyplot as plt

def column_to_pie(data, category):
    values = data[category].value_counts()
    labels, sizes = values.index.values, values
    plt.pie(sizes, labels=labels)
    plt.title(category)
    plt.show()

def parse_data(data, categories, show=False):
    for category in categories:
        data[category] = data[category].astype(str)
        if show:
            column_to_pie(data, category)

    return data

if __name__ == '__main__':
    train_data = pd.read_csv('training_set_VU_DM_2014_small.csv', delimiter = ',')
    test_data = pd.read_csv('test_set_VU_DM_2014_small.csv', delimiter = ',')

    # all column headers
    categories = list(train_data)
    # show em
    train_data = parse_data(train_data, categories, True)