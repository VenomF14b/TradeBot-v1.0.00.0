from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sklearn
import MetaTrader5 as mt5
import pyodbc
import alpaca_trade_api as tradeapi
import datetime as dt
import decimal
import time
import os
import tensorflow as tf
import subprocess
import ctypes
import signal
import sys

model = load_model(r"EURUSD/EURUSD.h5")

# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')


# Load data from database
query = f"SELECT TOP 1 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
data = []
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
    data.append(row)
cursor.close()

# Convert data to numpy array and reverse the order of the rows
data = np.array(data[::-1])
X_new = data[:, 1:5]
O_data = data[:, 1:5] 

print("X_new")
print(X_new)
print("Odata")
print(O_data)


#scaler = MinMaxScaler()
#X_new = scaler.fit_transform(X_new)
#O_data = scaler.fit_transform(O_data)
# Using z-score normalization
#X = zscore(X)
#Y = zscore(Y)

# Increment timestamp value by 60 seconds
#next_timestamp = last_row[0] + 60
next_timestamp = data[-1, 0] + 60

print(next_timestamp)

# Create new input data X_new
X_latest = np.array([[
    next_timestamp,  # Timestamp for the next time frame
    data[-1, 1],             # Placeholder value for open
    data[-1, 2],             # Placeholder value for high
    data[-1, 3],             # Placeholder value for low
    data[-1, 4]              # Placeholder value for close
]])

X_latest = X_latest[:, 1:] # Select all rows and columns 1 to 4 (inclusive

print(X_latest)

# Make a prediction on the new data point
Y_pred = model.predict(X_latest)
print("Prediction:", Y_pred[0])
Pred_Open = Y_pred[0,0]
Pred_High = Y_pred[0,1]
Pred_Low = Y_pred[0,2]
Pred_Close = Y_pred[0,3]
Last_Open = O_data[0,0]
Last_High = O_data[0,1]
Last_Low = O_data[0,2]
Last_Close = O_data[0,3]
# Inverse transform the predicted values to get actual scale
#Y_pred_actual = scaler.inverse_transform(Y_pred)
#O_data = scaler.inverse_transform(O_data)
#print("Prediction on trained data (actual):", Y_pred_actual[0])


# Do something with the predictions
Decision_Adjustor_Buy = 0.000025
Last_Close_Buy_Helper = Last_Close + Decision_Adjustor_Buy
if np.any(Pred_Close > Last_Close_Buy_Helper):
    print("Last_Close_Buy_Helper")
    print(Last_Close_Buy_Helper)
    print("Buy")   
    # Buy Code
    # open the script in a new terminal window
    subprocess.run(['python', 'EURUSD/trainedmodel/EURUSD_buy.py'])
 

else:
    Decision_Adjustor_Sell = -0.000025
    Last_Close_Sell_Helper = Last_Close + Decision_Adjustor_Sell
    if np.any(Pred_Close < Last_Close_Sell_Helper):
       print("Last_Close_Sell_Helper")
       print(Last_Close_Sell_Helper)
       print("Sell")
       # sell code here
       # open the script in a new terminal window
       subprocess.run(['python', 'EURUSD/trainedmodel/EURUSD_sell.py'])

    else:
        print("Do nothing")
        # do nothing code here
        time.sleep(10)
