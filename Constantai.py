import MetaTrader5 as mt5
import numpy as np
import tensorflow as tf
import pyodbc
import datetime as dt
import datetime
import decimal
import time
import tkinter as tk
import subprocess
import os
import ctypes
import signal
import sys
from tkinter import messagebox
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from keras.models import load_model



# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Load data from database
query = f"SELECT TOP 50 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00 ORDER BY timestamp DESC"
#query = "SELECT timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00 ORDER BY timestamp DESC"
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

# Get a list of all files in the directory that start with "trained_model_"
model_files = [f for f in os.listdir('.') if f.startswith('trained_model_')]
# Sort the list by name (which will put the latest model at the end of the list)
model_files.sort()
# Load the last file in the list (which should be the latest model)
latest_model_file = model_files[-1]
model = load_model(latest_model_file)

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=4,
          validation_data=(X_test, Y_test))

# Use the model to predict when to buy or sell
predictions_norm = model.predict(X)
print("Prediction on trained data:", predictions_norm[0])

# Extract the file name data of the model loaded and print on screen
model_name = latest_model_file.split(".")[0]
timestamp = model_name.split("_")[-1]
print(f"saved model from file: {latest_model_file}, created at {timestamp}")
model.save(f"trained_model_{timestamp}.h5")

time.sleep(10)
