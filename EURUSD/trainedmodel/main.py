import time
import subprocess
import pyodbc
import logging
import MetaTrader5 as mt5
import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt
#import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(filename='EURUSD.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
logging.info('\n''\nTraining Information')






# Ask user if they want to train a new model and overwrite old one
user_response = messagebox.askyesno("Train New Model", "Do you want to train a new model and overwrite the old one?")

userIn_passedtime = input("Data request range in days:")
if userIn_passedtime == "":
    userIn_passedtime = 7
    print(userIn_passedtime)
try:
    userIn_passedtime = int(userIn_passedtime)
except ValueError:
    print("Invalid input! Please enter an integer value.")
userIn_traineraiRowselector = input("Data size number of rows: ")
if userIn_traineraiRowselector == "":
    userIn_traineraiRowselector = 10080
    print(userIn_traineraiRowselector)
try:
    userIn_traineraiRowselector = int(userIn_traineraiRowselector)
except ValueError:
    print("Invalid input! Please enter an integer value.")
userIn_traineraiEpochs = input("Training Epochs size: ")
if userIn_traineraiEpochs == "":
    userIn_traineraiEpochs = 100
    print(userIn_traineraiEpochs)
try:
    userIn_traineraiEpochs = int(userIn_traineraiEpochs)
except ValueError:
    print("Invalid input! Please enter an integer value.")
userIn_traineraiBatchsize = input("Training batch size: ")
if userIn_traineraiBatchsize == "":
    userIn_traineraiBatchsize = 5
    print(userIn_traineraiBatchsize)
try:
    userIn_traineraiBatchsize = int(userIn_traineraiBatchsize)
except ValueError:
    print("Invalid input! Please enter an integer value.")

#Update adata
symbol = "EURUSD" #Symbol selector
passedtime = days=7 #Historical data time adjustor in days
#Trainerai
traineraiRowselector = userIn_traineraiRowselector
traineraiEpochs = userIn_traineraiEpochs
traineraiBatchsize = userIn_traineraiBatchsize
TaimodelS = "EURUSD/EURUSD.h5" #Model to save


if user_response:
    # Run the script to train a new model
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    symbol = symbol
    timeframe = mt5.TIMEFRAME_M1
    #cmd info
    print("Training the new model") # Training the new model
    print("Connection to MetaTrader5 successful")
    print(symbol,"timeframe = " + str(timeframe))
    #logging info
    logging.debug("Training the new model") # Training the new model
    logging.debug("Connection to MetaTrader5 successful")
    #logging.debug(symbol,"timeframe = " + str(timeframe))

    end_time = dt.datetime.now()    # Calculate start and end times
    end_time += dt.timedelta(hours=3)
    start_time = end_time - dt.timedelta(passedtime)   
    #cmd info
    print("Data Time Start = " + str(start_time))
    print("Data Time End = " + str(end_time))

    print("Getting historical data")    # Get historical data
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    rates = np.array(rates)

    start_time = end_time   # Update start and end times
    end_time = dt.datetime.now()

    conn = pyodbc.connect('Driver={SQL Server};'  # Establish a connection to the SQL Express database
                          'Server=VENOM-CLIENT\SQLEXPRESS;'
                          'Database=TRADEBOT;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()  # Write the data to the database
    for rate in rates:
        timestamp = int(rate[0])
        cursor.execute("SELECT COUNT(*) FROM EURUSDAdata WHERE timestamp = ?", (timestamp,))    # Check if timestamp already exists in the database
        count = cursor.fetchone()[0]
        if count == 0:
             # Write the data to the database
             values = [timestamp, float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7])]
             cursor.execute("INSERT INTO EURUSDAdata (timestamp, [open], high, low, [close], tick_volume, spread, real_volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
             print(values)
    conn.commit()
    #cmd info
    print("MT data is up to date")


    # Connect to the SQL Express database
    server = 'VENOM-CLIENT\SQLEXPRESS'
    database = 'NSAI'
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=VENOM-CLIENT\SQLEXPRESS;'
                          'Database=TRADEBOT;'
                          'Trusted_Connection=yes;')

    # Load data from adata
    #query = f"SELECT TOP ({traineraiRowselector}) timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
    #data = []
    #cursor = conn.cursor()
    #cursor.execute(query)
    #for row in cursor:
    #    data.append(row)
    #cursor.close()

    query = f"SELECT TOP ({traineraiRowselector}) timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
    data = []
    cursor = conn.cursor()
    cursor.execute(query)
    for row in cursor:
        converted_row = []
        for item in row:
            if isinstance(item, str):
                try:
                    converted_item = int(item)
                except ValueError:
                    converted_item = item
            else:
                converted_item = item
            converted_row.append(converted_item)
        data.append(converted_row)
    cursor.close()

    # Convert data to numpy array and reverse the order of the rows
    data = np.array(data[::-1])
    X = data[:-1, 1:5]  # timestamp, [open], high, low, [close]# remove the last row to avoid the roll-over issue
    Y = np.roll(data[:, 1:5], -1, axis=0)[:-1]  # remove the last row to avoid roll-over issue and shift the Y values by one time step to predict the next set of datapoints
    print("X")
    print(X)
    print("Y")
    print(Y)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y)

    print("X Normalized")
    print(X)
    print("Y Normalized")
    print(Y)

    # Reshape the data
    #X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    #Y = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))

    #print("X Reshape")
    #print(X)
    #print("Y Reshape")
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
        layers.Dense(12, activation="relu"),
        layers.Dense(36, activation="relu"),
        layers.Dense(108, activation="relu"),
        layers.Dense(648, activation="relu"),
        layers.Dense(108, activation="relu"),
        layers.Dense(36, activation="relu"),
        layers.Dense(12, activation="relu"),
        layers.Dense(4)
    ])

    model.compile(optimizer="adam", loss="mse")

    # Print the model summary
    print(model.summary())

    # Print the model weights and biases
    for layer in model.layers:
        weights, biases = layer.get_weights()
        print(layer.name, weights, biases)

    # Train the model
    model.fit(X_train, Y_train, epochs=traineraiEpochs, batch_size=traineraiBatchsize,
              validation_data=(X_test, Y_test))

    # Save the model
    model.save(TaimodelS)
    

# Continue running the script

# initialize the MT5 connection
if not mt5.initialize():
    print("Failed to initialize MT5 connection!")
    exit()

# Set the total runtime for the program
total_runtime = 21600  # seconds

# Set the interval between script runs
script_interval = 60  # seconds

# Set the maximum number of allowed trades
max_trades = 5

# Get the start time for the program
start_time = time.time()

# Start the loop to run the script
while True:
    # Check if the total runtime has elapsed
    if time.time() - start_time >= total_runtime:
        print("Program has ended.")
        break
        
    # Check if there are too many open trades
    trades = mt5.positions_get()
    if len(trades) >= max_trades:
        print("Maximum trades opened.")
        time.sleep(script_interval)
        continue
    print("Opened Trades")
    print(trades)
    # Run the script
    script_path = "EURUSD/trainedmodel/EURUSD_adata.py"
    subprocess.Popen(['python', script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)

    # Wait for the interval
    time.sleep(script_interval)

# shutdown MT5 connection
mt5.shutdown()