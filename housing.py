import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

def location(data):
    """Parse location indexes to an interger ranking scheme. """
    if data == "A":
        return 1
    elif data == "C":
        return 2
    elif data == "FV":
        return 3
    else:
        return 4

def split_data(data, p=0.5):
    """Split the data into a learn and test set."""
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    split_at = int(p*len(data))
    learn = shuffled_data[:split_at].reset_index(drop=True)
    test = shuffled_data[split_at:].reset_index(drop=True)
    return learn, test


def learn_linear_regression(data):
    """ Fit a linear regression through the datapoints."""
    regression = linear_model.LinearRegression()

    data_train = data[["LotArea", "YearBuilt", "MSSubClass", "OverallCond", "MSZoning"]]
    label_train = data["SalePrice"]

    regression.fit(data_train, label_train)
    
    return regression

def learn_polynomial(data):

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    import numpy as np
    model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                          ('linear', LinearRegression(fit_intercept=False))])
    # fit to an order-3 polynomial data
    x = np.arange(5)
    y = 3 - 2 * x + x ** 2 - x ** 3
    model = model.fit(x[:, np.newaxis], y)
    model.named_steps['linear'].coef_


if __name__ == '__main__':
    
    # Read the data for housing prices.
    data = pd.read_csv('housing_data.csv', delimiter = ',') 
    data["MSZoning"] = data["MSZoning"].apply(location)
    learn_data, test_data = split_data(data)
    
    # Create the linear regressor and predict.
    regressor = learn_linear_regression(learn_data)
    prediction = regressor.predict(test_data[["LotArea", "YearBuilt", "MSSubClass", "OverallCond", "MSZoning"]])
    
    #   
    print(mean_squared_error(prediction, test_data["SalePrice"]))
    print(mean_absolute_error(prediction, test_data["SalePrice"]))
    
    # Create the polynomial regressor and predict.
    regressor = learn_polynomial(learn_data)
    prediction = regressor.predict(test_data[["LotArea", "YearBuilt", "MSSubClass", "OverallCond", "MSZoning"]])
        
    print(mean_squared_error(prediction, test_data["SalePrice"]))
    print(mean_absolute_error(prediction, test_data["SalePrice"]))