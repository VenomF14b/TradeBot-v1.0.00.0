import MetaTrader5 as mt5
import numpy as np
import tensorflow as tf
import pyodbc
import datetime as dt
import decimal
import time
import tkinter as tk
import subprocess
from tkinter import messagebox
from tensorflow import keras
from tensorflow.keras import layers



bars = 1

# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Execute SQL query to get the latest record
#cursor = conn.cursor()
#cursor.execute("SELECT TOP 1 * FROM Tdata00 ORDER BY timestamp DESC")
#latest_record = cursor.fetchone()
#print(latest_record)




# Load data from database
#query = f"SELECT TOP 10 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00"
query = f"SELECT TOP 200 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00 ORDER BY timestamp DESC"
#query = "SELECT TOP 2000 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00 ORDER BY timestamp DESC"
data = []


cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
    data.append(row)
cursor.close()

#print(data)

# Convert data to numpy array and reverse the order of the rows
data = np.array(data[::-1])
X = data[:, 1:9]  # open
Y = data[:, 4:5]  # close

print("X")
print(X)
print("Y")
print(Y)

# Split the data into training and testing sets
split = int(0.95 * len(X))
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
    layers.Dense(64, activation="relu", input_shape=[len(X[0])]),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="tanh")
])
model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X_train, Y_train, epochs=1000, batch_size=32,
          validation_data=(X_test, Y_test))


# Use the model to predict when to buy or sell
predictions = model.predict(X)
print("Prediction on trained data:", predictions[0])

# Do something with the predictions
if np.any(predictions > 0.9):
    print("Buy")
elif np.any(predictions < -0.9):
    # sell code here
    print("Sell")
else:
    # do nothing code here
    print("Do nothing")

timestamp = int(time.time())  # get current timestamp

model.save(f"trained_model_{timestamp}.h5")  # save model with timestamp in the file name
    
# run the subprocess
process = subprocess.Popen(["python", "predict.py"])
# wait for the subprocess to complete
process.wait()
# resume the Tdata00.py script
print("predict has finished, resuming ai")







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

