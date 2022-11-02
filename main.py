import pandas as pd
from sklearn import tree

# Get train data from .csv file
train_data = pd.read_csv('./Data/train.csv')
# print(train_data.head())

# Get rid of unecessary data and establish dependent
train_features = train_data.drop('column_name', axis = 'columns')
train_price = train_data.SalePrice

# Create regression model and fit it to train data
regressor = tree.DecisionTreeRegressor()
regressor.fit(train_features, train_price)

# Get test data from .csv file
test_data = pd.read_csv("./Data/test.csv")
# print(test_data.head())

# Get rid of unecessary data
test_features = train_data.drop('column_name', axis = 'columns')

# Predict price of test data and print results
test_price = regressor.predict(test_features)
print(test_price)