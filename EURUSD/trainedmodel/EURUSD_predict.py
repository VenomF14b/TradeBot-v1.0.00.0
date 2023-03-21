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
query = f"SELECT TOP 2 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
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


scaler = MinMaxScaler()
X_new = scaler.fit_transform(X_new)
O_data = scaler.fit_transform(O_data)
# Using z-score normalization
#X = zscore(X)
#Y = zscore(Y)

print("X_Scaled")
print(X_new)
print("Y_Scaled")
print(O_data)

# Last row of original input data X
last_row = X_new[-1]

# Increment timestamp value by 60 seconds
next_timestamp = last_row[0] + 60

# Create new input data X_new
X_latest = np.array([[
    next_timestamp,  # Timestamp for the next time frame
    0.0,             # Placeholder value for open
    0.0,             # Placeholder value for high
    0.0,             # Placeholder value for low
    0.0              # Placeholder value for close
]])

X_latest = X_latest[:, 1:] # Select all rows and columns 1 to 4 (inclusive

# Make a prediction on the new data point
Y_pred = model.predict(X_latest)
print("Prediction on trained data (normalized):", Y_pred[0])

# Inverse transform the predicted values to get actual scale
Y_pred_actual = scaler.inverse_transform(Y_pred)
O_data = scaler.inverse_transform(O_data)
print("Prediction on trained data (actual):", Y_pred_actual[0])


# Do something with the predictions
Open_adjust_up = 0.00003
if np.any(Y_pred > O_data + Open_adjust_up):
    print("Y_pred_actual") 
    print(Y_pred)
    print("Buy")
    print("Open Price Adjustor", O_data + Open_adjust_up)    
    # Buy Code
    # open the script in a new terminal window
    script_path = "EURUSD/trainedmodel/EURUSD_buy.py"
    os.system(f"start cmd /k python {script_path}")

else:
    Open_adjust_down = -0.00003
    if np.any(Y_pred < O_data + Open_adjust_down):
       print("Sell")
       print("Open Price Adjustor", O_data + Open_adjust_down)
       # sell code here
       # open the script in a new terminal window
       script_path = "EURUSD/trainedmodel/EURUSD_sell.py"
       os.system(f"start cmd /k python {script_path}")

    else:
        print("Do nothing")
        # do nothing code here


# call the other script
subprocess.Popen(['python', 'EURUSD/trainedmodel/EURUSD_constantai.py'])
