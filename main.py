'''
I think that some types of values are present in train but not test so the dummies are inequal
train test is 50% split
'''

import pandas
from sklearn.ensemble import HistGradientBoostingRegressor as Regressor

# Get data
features = [pandas.read_csv('./Data/train.csv').drop('Id', axis='columns'),
            pandas.read_csv("./Data/test.csv").drop('Id', axis='columns')]
prices = [pandas.read_csv('./Data/train.csv')['SalePrice']]

# Create dummies
full_set = pandas.concat(features)
full_set = pandas.get_dummies(data=full_set, drop_first=False)

# Reseparate train and test
train_features = full_set[:1462,:]
test_features = full_set[1462:,:]

# Create and train model
model = Regressor()
model.fit(train_features.values, train_price.values)

# Predict price of test data and print results
test_prices = model.predict(test_features.values)

print(test_prices)