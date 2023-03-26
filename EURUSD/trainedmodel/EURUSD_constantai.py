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
import glob
from tkinter import messagebox
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from keras.models import load_model
import logging

logging.basicConfig(filename='EURUSD.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.info('\nTraining Information')

# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Load data from database
query = f"SELECT TOP 3 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
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
logging.debug("X")
logging.debug(X)
logging.debug("Y")
logging.debug(Y)

# Load data from database
query = f"SELECT TOP 3 position_ID, profit FROM EURUSDWLdata ORDER BY position_ID DESC"
profit_data = []
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
    profit_data.append(row)

print(profit_data)

# Convert data to numpy array and reverse the order of the rows
profit_data = np.array(profit_data[::-1])
print(profit_data)
profit = profit_data[:, 1]  # Select only the second column
print(profit)

cursor.close()

logging.debug("X")
logging.debug(X)
logging.debug("Y")
logging.debug(Y)
logging.debug("Profit")
logging.debug(profit)

# Calculate the reward based on the profit
reward = []
for i in range(len(profit) - 1):
    if profit[i] < 0:
        reward.append(-1)
    else:
        reward.append(1)
reward.append(0)
reward = np.array(reward)
reward = reward[:]
logging.info("Reward")
logging.debug(reward)

# Split the data into training and testing sets
split = int(0.70 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
R_train, R_test = reward[:split], reward[split:]

logging.debug("X_train")
logging.debug(X_train)
logging.debug("X_test")
logging.debug(X_test)
logging.debug("Y_train")
logging.debug(Y_train)
logging.debug("Y_test")
logging.debug(Y_test)
logging.debug("R_train")
logging.debug(R_train)
logging.debug("R_test")
logging.debug(R_test)

# Get a list of all the model files in the directory
#file_list = glob.glob("EURUSD/EURUSD_*.h5")

# Sort the file list by timestamp in descending order
#file_list.sort(key=os.path.getmtime, reverse=True)

# Load the latest model
#model = load_model(file_list[0])
model = load_model("EURUSD/EURUSD.h5")

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=1, sample_weight=R_train,
          validation_data=(X_test, Y_test), validation_steps=len(X_test),
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)])


# Use the model to predict when to buy or sell
#predictions_norm = model.predict(X)
#logging.debug("Prediction on trained data:", predictions_norm[0])

#model.save(file_list[0])
model.save("EURUSD/EURUSD.h5")

subprocess.run(['python', 'EURUSD/trainedmodel/EURUSD_predict.py'])

