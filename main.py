# https://github.com/alex-tanton/House-Pricing-Competition

import pandas
from sklearn.ensemble import HistGradientBoostingRegressor as Regressor

# Get data
features = [pandas.read_csv('./Data/train.csv').drop('Id', axis='columns').drop('SalePrice', axis='columns'),
            pandas.read_csv("./Data/test.csv").drop('Id', axis='columns')]
prices = [pandas.read_csv('./Data/train.csv')['SalePrice']]

# Create dummies
full_set = pandas.concat(features)
full_set = pandas.get_dummies(data=full_set, drop_first=False)

# Reseparate train and test
train_features = full_set[:1460]
test_features = full_set[1460:]

# Create and train model
model = Regressor()
model.fit(train_features.values, prices[0].values)

# Predict price of test data and print results
test_prices = model.predict(test_features.values)

output = pandas.DataFrame({
    "Id" : pandas.read_csv("./Data/test.csv")['Id'],
    "SalePrice" : test_prices
})

output.to_csv("submission.csv", index=False)