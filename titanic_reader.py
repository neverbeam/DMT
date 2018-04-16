import pandas as pd

if __name__ == '__main__':
    train_data = pd.read_csv('titanic_train.csv', delimiter = ',')
    print(train_data)