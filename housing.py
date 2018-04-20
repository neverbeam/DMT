import pandas as pd
from sklearn import linear_model,svm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

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

    data_train = data[["LotArea", "YearBuilt", "MSSubClass", "OverallCond","BedroomAbvGr","TotRmsAbvGrd", "FullBath","BsmtFullBath", "GarageArea", "OverallQual", "TotalBsmtSF", "YrSold"]]
    label_train = data["SalePrice"]

    regression.fit(data_train, label_train)
    
    return regression

def learn_SVR(data):
    """ Fit a support vector regression through the datapoints."""
    regression = svm.SVR()

    data_train = data[["LotArea", "YearBuilt", "MSSubClass", "OverallCond","BedroomAbvGr","TotRmsAbvGrd", "FullBath","BsmtFullBath", "GarageArea", "OverallQual", "TotalBsmtSF", "YrSold"]]
    label_train = data["SalePrice"]

    regression.fit(data_train, label_train)
    
    return regression

if __name__ == '__main__':
    
    # Read the data for housing prices.
    data = pd.read_csv('housing_data.csv', delimiter = ',') 
    learn_data, test_data = split_data(data)
    
    # Create the linear regressor and predict.
    regressor = learn_linear_regression(learn_data)
    prediction = regressor.predict(test_data[["LotArea", "YearBuilt", "MSSubClass", "OverallCond", "BedroomAbvGr","TotRmsAbvGrd","FullBath", "BsmtFullBath", "GarageArea", "OverallQual", "TotalBsmtSF", "YrSold"]])
   
    print(mean_squared_error(prediction, test_data["SalePrice"]))
    print(mean_absolute_error(prediction, test_data["SalePrice"]))
    
    #Create the Support vector regressor and predict.
    regressor = learn_SVR(learn_data)
    prediction = regressor.predict(test_data[["LotArea", "YearBuilt", "MSSubClass", "OverallCond", "BedroomAbvGr","TotRmsAbvGrd","FullBath", "BsmtFullBath", "GarageArea", "OverallQual", "TotalBsmtSF", "YrSold"]])
        
    print(mean_squared_error(prediction, test_data["SalePrice"]))
    print(mean_absolute_error(prediction, test_data["SalePrice"]))
    