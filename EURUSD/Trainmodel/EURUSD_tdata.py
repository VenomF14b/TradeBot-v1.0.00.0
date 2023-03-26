import MetaTrader5 as mt5
import numpy as np
import tensorflow as tf
import tkinter as tk
import datetime as dt
import decimal
import pyodbc
import subprocess
import time
import ctypes
import signal
import sys
import os
from tensorflow import keras
from tensorflow.keras import layers
from tkinter import messagebox
from datetime import datetime

print("Establising Connection to MT5, Please wait") # Connect to MT5
mt5.initialize()
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1
print("Connection Successful")
print(symbol,"timeframe = " + str(timeframe))

end_time = dt.datetime.now()    # Calculate start and end times
start_time = end_time - dt.timedelta(days=30)   
print("Data Time Start = " + str(start_time))
print("Data Time End = " + str(end_time))

print("Getting historical data")    # Get historical data
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
rates = np.array(rates)

start_time = end_time   # Update start and end times
end_time = dt.datetime.now()

print("Establishing a connection to the SQL Express database")  # Establish a connection to the SQL Express database
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')
print("Connection established")
print("database update in progress please wait...")
cursor = conn.cursor()  # Write the data to the database
for rate in rates:
    timestamp = int(rate[0])
    cursor.execute("SELECT COUNT(*) FROM EURUSDTdata WHERE timestamp = ?", (timestamp,))    # Check if timestamp already exists in the database
    count = cursor.fetchone()[0]
    if count == 0:
         # Write the data to the database
         values = [timestamp, float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7])]
         cursor.execute("INSERT INTO EURUSDTdata (timestamp, [open], high, low, [close], tick_volume, spread, real_volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
         print(values)
conn.commit()
print("SQL complete MT data is up to date")


# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Load data from database
query = f"SELECT TOP 1440 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDTdata ORDER BY timestamp DESC"
#query = "SELECT timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDTdata ORDER BY timestamp DESC"
data = []
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
    data.append(row)
cursor.close()

# Convert data to numpy array and reverse the order of the rows
data = np.array(data[::-1])
X = data[:, 1:5]  # timestamp, [open], high, low, [close]

# Shift the Y values by one time step to predict the next set of datapoints
Y = np.roll(data[:, 1:5], -1, axis=0)
print("X")
print(X)
print("Y")
print(Y)

# Split the data into training and testing sets
split = int(0.70 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

# Define the AI model
model = keras.Sequential([
    layers.Dense(4, activation="relu", input_shape=[len(X[0])]),
    layers.Dense(8, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(4)
])

model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X_train, Y_train, epochs=20, batch_size=1,
          validation_data=(X_test, Y_test))

# Use the model to predict when to buy or sell
predictions_norm = model.predict(X)
print("Prediction on trained data:", predictions_norm[0])

# Generate a timestamp string
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Define the file path with the timestamp
file_path = f"EURUSD/EURUSD.h5"

# Save the model with the timestamp in the file name
model.save(file_path)    

script_path = "EURUSD/trainedmodel/EURUSD_adata.py"
subprocess.call(['python', script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
time.sleep(10)
