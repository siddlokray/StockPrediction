# import libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
import matplotlib.pyplot as plt
import yfinance as yf

# get stock data
ticker = 'AAPL'
data = yf.Ticker(ticker)
data = data.history(period='1d', start='2019-01-01')

# get data for plot
plot = yf.Ticker(ticker)
plot = plot.history(period='1d', start='2016-01-01')

# edit data
data = data.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis=1)
plot = plot.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis=1)
data = np.array(data)

# format data
X = []
Y = []

for i in range(len(data)):
    if i+7 == len(data):
         break
    else:
        x = [data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5], data[i+6]]
        x = np.array(x)
        y = [data[i+7]]
        y = np.array(y)
    X.append(x)
    Y.append(y)

# set train test and validation data
X_train = X
y_train = Y

X_train = np.array(X_train).reshape(-1,7,1)
y_train = np.array(y_train)

# create model
model = Sequential()

# input layer
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(7,1)))
model.add(Dropout(0.05))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(1))

# compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# fit model
model.fit(X_train, y_train, epochs=5, verbose=2)

# predict future
future_days = 30

for i in range(future_days):
    test = [data[-7], data[-6], data[-5], data[-4], data[-3], data[-2], data[-1]]
    test = np.array(test).reshape(-1,7,1)
    prediction = model.predict(test)
    data = np.concatenate((data, prediction), axis=0)
    prediction = prediction.reshape(-1,1,1)
    prediction = prediction.tolist()
    prediction = [item for sublist in prediction for item in sublist]
    prediction = [item for sublist in prediction for item in sublist]
    for i in prediction:
        prediction = i
    plot = plot.append([{'Close':prediction}], ignore_index=True)

# plot data
plt.plot(plot['Close'][:-30], c='c', label='Closing Price')
plt.plot(plot['Close'][-30:], c='lightpink', label='Future Predictions')
plt.title(ticker + ' Predicted Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
