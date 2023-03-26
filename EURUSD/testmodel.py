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

# Convert data to numpy array and reverse the order of the rows
profit_data = np.array(profit_data[::-1])
profit = profit_data[:, 1]  # Select only the second column

cursor.close()

print("X")
print(X)
print("Y")
print(Y)
print("Profit")
print(profit)

# Calculate the reward based on the profit
reward = []
for i in range(len(profit) - 1):
    if profit[i] < 0:
        reward.append(-1)
    else:
        reward.append(1)
reward.append(0)
reward = np.array(reward)
reward = reward[:-1]
print(reward)

# Split the data into training and testing sets
split = int(0.80 * len(X))
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

# Get a list of all the model files in the directory
#file_list = glob.glob("EURUSD/EURUSD_*.h5")

# Sort the file list by timestamp in descending order
#file_list.sort(key=os.path.getmtime, reverse=True)

# Load the latest model
#model = load_model(file_list[0])
model = load_model("EURUSD/EURUSD.h5")

# Train the model
model.fit(X_train, Y_train, epochs=5, batch_size=1, sample_weight=R_train,
          validation_data=(X_test, Y_test), validation_steps=len(X_test),
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)])


# Use the model to predict when to buy or sell
#predictions_norm = model.predict(X)
#logging.debug("Prediction on trained data:", predictions_norm[0])

#model.save(file_list[0])
#model.save("EURUSD/EURUSD.h5")

subprocess.run(['python', 'EURUSD/trainedmodel/EURUSD_predict.py'])







#import numpy as np
#import MetaTrader5 as mt5
#import tensorflow as tf
#import pyodbc
#import datetime as dt
#import datetime
#import decimal
#import time
#import tkinter as tk
#import subprocess
#import os
#import ctypes
#import signal
#import sys
#import glob
#from tkinter import messagebox
#from tensorflow import keras
#from tensorflow.keras import layers
#from datetime import datetime
#from keras.models import load_model
#import logging

#logging.basicConfig(filename='EURUSD.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
#logging.info('\nTraining Information')

# Connect to the SQL Express database
#server = 'VENOM-CLIENT\SQLEXPRESS'
#database = 'NSAI'
#conn = pyodbc.connect('Driver={SQL Server};'
#                      'Server=VENOM-CLIENT\SQLEXPRESS;'
#                      'Database=TRADEBOT;'
#                      'Trusted_Connection=yes;')

# Load data from database
#query = f"SELECT TOP 3 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
#query = "SELECT timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
#data = []
#cursor = conn.cursor()
#cursor.execute(query)
#for row in cursor:
#    data.append(row)
#cursor.close()

# Convert data to numpy array and reverse the order of the rows
#data = np.array(data[::-1])
#X = data[:, 1:5]  # timestamp, [open], high, low, [close]

# Shift the Y values by one time step to predict the next set of datapoints
#Y = np.roll(data[:, 1:5], -1, axis=0)
#logging.debug("X")
#logging.debug(X)
#logging.debug("Y")
#logging.debug(Y)


# Split the data into training and testing sets
#split = int(0.80 * len(X))
#X_train, X_test = X[:split], X[split:]
#Y_train, Y_test = Y[:split], Y[split:]

#logging.debug("X_train")
#logging.debug(X_train)
#logging.debug("X_test")
#logging.debug(X_test)
#logging.debug("Y_train")
#logging.debug(Y_train)
#logging.debug("Y_test")
#logging.debug(Y_test)

# Create a label array indicating whether each trade was a win or a loss
#buy_data = data[:, 1]
#sell_data = data[:, 4]
#label_array = np.where(sell_data > buy_data, 1, -1)

# Define a reward/punishment array based on the label array
#reward_array = np.where(label_array == 1, 1, -1)

# Get a list of all the model files in the directory
#file_list = glob.glob("EURUSD/EURUSD_*.h5")
# Sort the file list by timestamp in descending order
#file_list.sort(key=os.path.getmtime, reverse=True)
# Load the latest model
#model = load_model(file_list[0])
#model = load_model("EURUSD/EURUSD.h5")


# Train the model with reward/punishment
#model.fit(X_train, Y_train, epochs=5, batch_size=1,
#          validation_data=(X_test, Y_test), sample_weight=reward_array)

# Use the model to predict when to buy or sell
#predictions_norm = model.predict(X)
#logging.debug("Prediction on trained data:", predictions_norm[0])

#model.save(file_list[0])
#model.save("EURUSD/EURUSD.h5")

#subprocess.run(['python', 'EURUSD/trainedmodel/EURUSD_predict.py'])

# Define some example buy and sell trade data
#buy_data = [100, 120, 80, 90, 110]
#sell_data = [105, 110, 85, 95, 115]

# Create a label array indicating whether each trade was a win or a loss
#label_array = np.where(np.array(sell_data) > np.array(buy_data), 1, -1)

# Define a reward/punishment array based on the label array
#reward_array = np.where(label_array == 1, 1, -1)

# Define your model
#class MyModel:
#    def __init__(self):
        # Your model initialization code here
#        pass
    
#    def predict(self, input_data):
        # Your model prediction code here
#        pass
    
#    def update_weights(self, input_data, target_data, reward_data):
        # Your model weight update code here
        # Add the reward/punishment signal to the target data
#        target_data *= reward_data
        # Your weight update code
#        pass

# Create an instance of your model
#my_model = MyModel()

# Define your training loop
#for i in range(num_epochs):
#    # Get the input data for this epoch
#    input_data = get_input_data()

#    # Use your model to make predictions
#    predicted_data = my_model.predict(input_data)

#    # Get the target data for this epoch (e.g., the expected output)
#    target_data = get_target_data




# Train the model on new data
model.fit(X_train_new, Y_train_new, epochs=5, batch_size=1, sample_weight=R_train_new,
          validation_data=(X_test_new, Y_test_new), validation_steps=len(X_test_new),
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)])

# Train the model
model.fit(X_train, Y_train, epochs=5, batch_size=1, sample_weight=R_train,
          validation_data=(X_test, Y_test), validation_steps=len(X_test),
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)])