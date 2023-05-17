import sys
import os

sys.path.insert(0, os.getcwd())

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from lazypredict.Supervised import LazyRegressor
import matplotlib.pyplot as plt

from NovabiomPredUtils import NovaPredictTools as npt
from sklearn.metrics import classification_report

import zipfile
import warnings

import xgboost as xgb

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\1665865\Documents\practice_ITMO_progect_GPN\data_for_predict.csv', header=0)
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
y = df['PDSC 210С, мин']
x = df.drop(['PDSC 210С, мин'], axis = 1)
s = npt.Lazyregressor_vae(x, y)
print(s.head(10))

import xgboost as xgb

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Create linear regression object
scaler = MinMaxScaler()
scaler.fit(X_train)
sc_x_train = scaler.transform(X_train)
scaled_test_x_ = scaler.transform(X_test)
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(sc_x_train, Y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(scaled_test_x_)

# # The coefficients
# # print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(Y_test, diabetes_y_pred)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, diabetes_y_pred))
print(X_test)
# Plot outputs
plt.figure(facecolor = "white")
plt.scatter(Y_test, diabetes_y_pred, color="black")
plt.xlabel('PDSC 210С real', fontsize=11)
plt.ylabel('PDSC 210С predicted', fontsize=11)
plt.show()

xgbr = xgb.XGBRegressor(verbosity=0) 
xgbr.fit(sc_x_train, Y_train)
score = xgbr.score(sc_x_train, Y_train)  
print("Training score: ", score)
ypred = xgbr.predict(scaled_test_x_)

print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(Y_test, ypred)))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(Y_test, ypred))
# print(X_test)
# Plot outputs
plt.figure(facecolor = "white")
plt.scatter(Y_test, ypred, color="black")
plt.xlabel('PDSC 210С real', fontsize=11)
plt.ylabel('PDSC 210С predicted', fontsize=11)
plt.show()