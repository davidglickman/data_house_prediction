# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 19:23:37 2023

@author: glick
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# dataframe
df = pd.read_csv("kc_house_data.csv")
df.dtypes
df.describe()

# drop ID from the dataframe
df.drop('id',axis=1, inplace = True)
# df.drop(['id', 'Unnamed: 0'], axis=1, inplace = True)
df.dtypes

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

# replace missing values with mean
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

numberOfFloors = df['floors'].value_counts().to_frame()

# present correlation between variables 

sns.boxplot(data=df, x="waterfront", y="price")

sns.regplot(data = df, x = "sqft_above", y = "price")

df.corr()['price'].sort_values()

# predict value price using linear regression 'long' variable based

X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

# predict value price using linear regression 'sqft_living' variable based

X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)
predictedY = lm.predict(X)

# R score calculation
rScore = r2_score(Y,predictedY)

# predict "price" with a list of features
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
lm = LinearRegression()
onlyFeaturesDf = df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
lm.fit(onlyFeaturesDf, Y)
predictedY = lm.predict(onlyFeaturesDf)

# R score calculation
rScore = r2_score(Y,predictedY)


# create pipeline using sklearn pipeline object
Input=(('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression()))
pipe = Pipeline(Input)
pipe.fit(onlyFeaturesDf, Y)
predictedY = pipe.predict(onlyFeaturesDf)
rScore = r2_score(Y,predictedY)

# Split training and testing
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

# Implement ridge regression
ridgeRegressionEstimator = Ridge(alpha=0.1)
ridgeRegressionEstimator.fit(x_train, y_train)
ridgeRegressionEstimatorTestSet = ridgeRegressionEstimator.predict(x_test)
rScore = r2_score(y_test,ridgeRegressionEstimatorTestSet)
print (rScore)

# Implement ridge regression on second order polynomial transform
poly = PolynomialFeatures(2)
secondOrderTransformX_train = poly.fit_transform(x_train)
ridgeRegressionEstimator = Ridge(alpha=0.1)
ridgeRegressionEstimator.fit(secondOrderTransformX_train, y_train)
secondOrderTransformX_test = poly.fit_transform(x_test)
ridgeRegressionEstimatorTestSetSecondOrder = ridgeRegressionEstimator.predict(secondOrderTransformX_test)
rScore = r2_score(y_test,ridgeRegressionEstimatorTestSetSecondOrder)


