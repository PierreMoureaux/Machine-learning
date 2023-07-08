# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:07:43 2023

@author: pmoureaux"""

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score

#A very simple and vanilla bullet fixed-rate TRS class
class TRS:
    
    def __init__(self, S, K, T, r, r_TRS,Type):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.r_TRS = r_TRS
        self.Type = Type
        self.mv = self.mv()
    
    def mv(self):
        mv_TRS = self.S - math.exp(-self.r *self.T)*(self.K + self.r_TRS*self.S*self.T)
        if self.Type == "Receive performance":
            return mv_TRS
        if self.Type == "Pay performance":
            return -mv_TRS

#dataset
r = np.arange(.0, .1, .01) #interest rates
Strike = np.arange(50, 155, 5) #strike price (aka settlement price)
T = np.arange(0.1, 2.1, 0.1) #time to maturity
r_TRS = np.arange(.0, .2, .02) #TRS financing rate (aka repo rate)

data = []
for r_ in r:
    for Strike_ in Strike:
        for T_ in T:
            for r_TRS_ in r_TRS:
                data.append([r_, Strike_, T_, r_TRS_, \
                             TRS(100, Strike_, T_, r_, r_TRS_, "Receive performance").mv])
data = np.asarray(data)

#training and test datasets
X = data[:,:4] #params r, strike, T, r_TRS
y = data[:,4:5] #TRS market value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#ANN with four layers, 10 neurons each
#activation function: ReLU
ANN = Sequential()
ANN.add(Dense(10,input_dim = 4, activation = 'relu'))
ANN.add(Dense(10, activation = 'relu'))
ANN.add(Dense(10, activation = 'relu'))
ANN.add(Dense(10, activation = 'relu'))         
ANN.add(Dense(1))

#Loss function = MSE, optimizer: Adam
ANN.compile(loss = 'mean_squared_error', optimizer='adam')
# fit the ANN on the training dataset
ANN.fit(X_train, y_train, epochs = 150, batch_size = 16)

#prediction
y_pred = ANN.predict(X_test)

#Comparison real values and predictions on test dataset
plt.figure(figsize = (15,10))
plt.scatter(y_test, y_pred)
plt.xlabel("Real Value")
plt.ylabel("ANN Value")
plt.annotate("r-squared = {:.3f}".format(r2_score(y_test, y_pred)), (20, 1), size = 15)
plt.show()

K = 120 #strike price
r = 0.05 #risk-free interest rate
r_TRS = 0.07 #TRS financing rate
T = .5 #time to maturity
S = np.arange(50, 151, 1) #asset prices

PriceTheo = [TRS(S_, K, T, r, r_TRS, "Receive performance").mv for S_ in S]
PriceANN = [S_ / 100 * \
            ANN.predict(np.array([[r, K / S_ * 100, T, r_TRS]]))[0][0] for S_ in S]
    
#Comparison BS vs ANN prices
plt.figure(figsize = (15,10))
plt.plot(S, PriceTheo, label = "Theoretical market value")
plt.plot(S, PriceANN, label = "ANN market value")
plt.xlabel("Asset price")
plt.ylabel("TRS market value")
plt.show();