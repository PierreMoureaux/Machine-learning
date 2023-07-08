# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:48:31 2023

@author: pmoureaux
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from tensorflow import keras

# 1 - Datas
SOFR = pd.read_excel(r'C:\Users\moure\Securities finance derivatives\Securities finance - research and specific features\6 - Repo rate curve - machine learning\SOFR.xlsx').set_index('Effective Date')[::-1]
SOFR.index = pd.to_datetime(SOFR.index)
SOFR['days_from_start'] = (SOFR.index - SOFR.index[0]).days;
x = SOFR['days_from_start'].values.reshape(-1, 1)
y = SOFR['Rate (%)'].values

# 2 - Polynomial regression
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(x)
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)
y_predicted = poly_reg_model.predict(poly_features)
print(np.square(np.subtract(y,y_predicted)).mean())

fig, ax = plt.subplots(figsize=(10, 6))
plt.title("SOFR Polynomial regression", size=16)
ax.plot(x, y, label="Actual SOFR")
ax.plot(x, y_predicted,'r', label="predicted SOFR")
ax.legend()
plt.show()

# 3 - Neural network regression
model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model.add(keras.layers.Dense(units = 64, activation = 'elu'))
model.add(keras.layers.Dense(units = 64, activation = 'elu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))
model.compile(loss='mse', optimizer="adam")

model.fit(x, y, epochs=200, verbose=1)
y_predicted2 = model.predict(x)
print(np.square(np.subtract(y,y_predicted2)).mean())

fig, ax = plt.subplots(figsize=(10, 6))
plt.title("SOFR Neural network regression", size=16)
ax.plot(x, y, label="Actual SOFR")
ax.plot(x, y_predicted2,'r', label="predicted SOFR")
ax.legend()
plt.show()