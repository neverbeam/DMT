import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_date_time(value):
    new_val = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    return new_val

def column_to_pie(data, category):
    values = data[category].value_counts()
    labels, sizes = values.index.values, values
    plt.pie(sizes, labels=labels)
    plt.title(category)
    plt.show()

def parse_data(data, categories, show=False):
    for category in categories:
        # data[category] = data[category].astype(str)
        if show:
            column_to_pie(data, category)

    data["date_time"] = data["date_time"].apply(parse_date_time)
    
    return data

def stripcompetition(item):
    if item == "nan":
        return 0
    else:
        return item
    

def changecompetition(data, categories):
    for category in categories[-24:-3]:
        data[category] = data[category].apply(stripcompetition)
        
    return data

if __name__ == '__main__':
    train_data = pd.read_csv('training_set_VU_DM_2014_small.csv', delimiter = ',')
    test_data = pd.read_csv('test_set_VU_DM_2014_small.csv', delimiter = ',')

    # all column headers
    categories = list(train_data)
    # show em
    train_data = parse_data(train_data, categories)

    train_data = changecompetition(train_data, categories)
    print (train_data["comp2_rate"])
    