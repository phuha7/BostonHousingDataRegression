#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:52:33 2022

@author: phuha

R2 for Multiple Linear Regression: 0.589
R2 for Decision Tree Regression: 0.601
R2 for Random Forest Regression: 0.742
R2 for XGBoost: 0.737
"""

# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('/Users/phuha/Desktop/boston_housing.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# %% Training multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %% Training the Decision Tree Regression model on training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)



# %% Training the Random Forest Regression model on training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# %% Training the dataset with xgboost
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)

# %% Predicting the Test set Results
y_pred = regressor.predict(X_test)

# evaluating model performance
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
