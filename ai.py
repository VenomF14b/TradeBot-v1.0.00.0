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
#from sklearn.preprocessing import MinMaxScaler
#from scipy.stats import zscore
from tkinter import messagebox
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime


# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Load data from database
query = f"SELECT TOP 300 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00 ORDER BY timestamp DESC"
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



# Normalize the data 
# Using MinMax normalization
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)
#Y = scaler.fit_transform(Y)
# Using z-score normalization
#X = zscore(X)
#Y = zscore(Y)

#print("X_Scaled")
#print(X)
#print("Y_Scaled")
#print(Y)

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
    layers.Dense(4, activation="linear")
    #layers.Dense(1, activation="tanh")

    #layers.Dense(5, activation="relu", input_shape=[len(X[0])]),
    #layers.Dense(10, activation="relu"),
    #layers.Dense(40, activation="relu"),
    #layers.Dense(80, activation="relu"),
    #layers.Dense(40, activation="relu"),
    #layers.Dense(10, activation="relu"),
    #layers.Dense(5, activation="linear")
    #layers.Dense(1, activation="tanh")
])


model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=4,
          validation_data=(X_test, Y_test))


# Use the model to predict when to buy or sell
predictions_norm = model.predict(X)
print("Prediction on trained data:", predictions_norm[0])

# Inverse transform the predicted values to get actual scale
#predictions_actual = scaler.inverse_transform(predictions_norm)
#print("Prediction on trained data (actual):", predictions_actual[0])

timestamp = int(time.time())  # get current timestamp

model.save(f"trained_model_{timestamp}.h5")  # save model with timestamp in the file name
    








# Define the reward function for reinforcement learning
#def get_reward(action, next_state):
#    reward = np.sum(np.sign(action) == np.sign(next_state))
#    return reward

# Use reinforcement learning to improve the model
#gamma = 0.95  # discount factor
#for i in range(bars - 1):
#    state = X[i]
#    action = model.predict(np.array([state]))
#    next_state = X[i + 1]
#    reward = get_reward(action, next_state[-1])
#   target = reward + gamma * np.max(model.predict(np.array([next_state])))
#    target_f = model.predict(np.array([state]))
#    target_f[0][np.argmax(action)] = target
#    model.fit(np.array([state]), target_f, epochs=10, verbose=0)

# Use the improved model to make predictions
#predictions = model.predict(X)
#print("Pediction on reinforcement" , predictions)

