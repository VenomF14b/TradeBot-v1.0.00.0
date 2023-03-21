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
query = f"SELECT TOP 64 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
#query = "SELECT timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
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

model = load_model(r"EURUSD/EURUSD.h5")

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=2,
          validation_data=(X_test, Y_test))

# Use the model to predict when to buy or sell
predictions_norm = model.predict(X)
print("Prediction on trained data:", predictions_norm[0])

model.save(r"EURUSD/EURUSD.h5")

subprocess.run(['python', 'EURUSD/trainedmodel/EURUSD_predict.py'])

