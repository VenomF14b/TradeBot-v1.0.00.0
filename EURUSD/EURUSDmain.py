import time
import subprocess
import win32con
import pyodbc
import logging
import MetaTrader5 as mt5
import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
import configparser
from tensorflow import keras
from tensorflow.keras import layers
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create logger for EURUSDmodel
EURUSDmodellog = logging.getLogger('EURUSDmodel')
file_handler1 = logging.FileHandler('EURUSD\EURUSDmodel.log')
file_handler1.setFormatter(formatter)
EURUSDmodellog.addHandler(file_handler1)
EURUSDmodellog.setLevel(logging.DEBUG)
logging.info('\n''\n')

# Create logger for EURUSDtdata
EURUSDtdatalog = logging.getLogger('EURUSDtdata')
file_handler2 = logging.FileHandler('EURUSD\EURUSDtdata.log')
file_handler2.setFormatter(formatter)
EURUSDtdatalog.addHandler(file_handler2)
EURUSDtdatalog.setLevel(logging.DEBUG)

# Create logger for EURUSDtparam
EURUSDtparamlog = logging.getLogger('EURUSDtparam')
file_handler3 = logging.FileHandler('EURUSD\EURUSDtparam.log')
file_handler3.setFormatter(formatter)
EURUSDtparamlog.addHandler(file_handler3)
EURUSDtparamlog.setLevel(logging.DEBUG)

# Create logger for EURUSDconparam
EURUSDconparamlog = logging.getLogger('EURUSDconparam')
file_handler4 = logging.FileHandler('EURUSD\EURUSDconparam.log')
file_handler4.setFormatter(formatter)
EURUSDconparamlog.addHandler(file_handler4)
EURUSDconparamlog.setLevel(logging.DEBUG)





# Read the configuration file, Script variables************************************************************************************
#**********************************************************************************************************************************
config = configparser.ConfigParser()
config.read('EURUSD/configEURUSD.ini')
training_params = config['Training Parameters']

#Update adata
symbol = "EURUSD" #Symbol selector
passedtime = days= + int(training_params.get('passedtime', '7')) #Historical data time adjustor in days
#Trainerai
MainTrain = False
traineraiRowselector = int(training_params.get('trainerairowselector', '10080'))
traineraiEpochs = int(training_params.get('traineraiepochs', '100'))
traineraiBatchsize = int(training_params.get('traineraibatchsize', '5'))
TraineraiSplit = 0.90
TaimodelS = "EURUSD/EURUSD.h5" #Model to save
#Program
TotalRuntime = 21600  # seconds
ScriptInterval = 60  # seconds
MaxTrades = 5
#**********************************************************************************************************************************

# Training the new model code *****************************************************************************************************
#**********************************************************************************************************************************
# MT5 Initialize, Logging infomation **********************************************************************************************
#**********************************************************************************************************************************
if MainTrain:
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    symbol = symbol
    timeframe = mt5.TIMEFRAME_M1
    EURUSDtdatalog.debug("Connection to MetaTrader5 successful")
    #**********************************************************************************************************************************

    # Calculate the start and end times, Logging information, Update start and end times **********************************************
    #**********************************************************************************************************************************
    end_time = dt.datetime.now()
    end_time += dt.timedelta(hours=3)
    start_time = end_time - dt.timedelta(passedtime)
    EURUSDtdatalog.debug("Data Time Start = " + str(start_time))
    EURUSDtdatalog.debug("Data Time End = " + str(end_time))
    EURUSDtdatalog.debug("Getting historical data")
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    rates = np.array(rates)
    start_time = end_time
    end_time = dt.datetime.now()
    #**********************************************************************************************************************************

    # Establish a connection to the database, Check if timestamp already exists, Writes the data to the database, Cmd information *****
    #**********************************************************************************************************************************
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=VENOM-CLIENT\SQLEXPRESS;'
                          'Database=TRADEBOT;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    for rate in rates:
        timestamp = int(rate[0])
        cursor.execute("SELECT COUNT(*) FROM EURUSDAdata WHERE timestamp = ?", (timestamp,))
        count = cursor.fetchone()[0]
        if count == 0:
             # Write the data to the database
             values = [timestamp, float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7])]
             cursor.execute("INSERT INTO EURUSDAdata (timestamp, [open], high, low, [close], tick_volume, spread, real_volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
             print(values)
    conn.commit()
    cursor.close()
    conn.close()
    print("MT data is up to date")
    #**********************************************************************************************************************************

    # Connect to the SQL Express database, Selects data to use for training ***********************************************************
    #**********************************************************************************************************************************
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=VENOM-CLIENT\SQLEXPRESS;'
                          'Database=TRADEBOT;'
                          'Trusted_Connection=yes;')
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
    conn.close()
    #**********************************************************************************************************************************

    # Data to numpy array reverse row order, X select remove last row, Y select remove last row shift values, Logging information *****
    #**********************************************************************************************************************************
    data = np.array(data[::-1])
    X = data[:-1, 1:5]
    Y = np.roll(data[:, 1:5], -1, axis=0)[:-1]
    X = X.astype('float32')
    Y = Y.astype('float32')
    EURUSDtdatalog.debug("X")
    EURUSDtdatalog.debug(X)
    EURUSDtdatalog.debug("Y")
    EURUSDtdatalog.debug(Y)
    #**********************************************************************************************************************************

    # Normalize the data
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #X = scaler.fit_transform(X)
    #Y = scaler.fit_transform(Y)

    # Split, Logging information, Define the AI model, Train the model, Saves model ***************************************************
    #**********************************************************************************************************************************
    split = int((TraineraiSplit) * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    EURUSDtdatalog.debug("X_train")
    EURUSDtdatalog.debug(X_train)
    EURUSDtdatalog.debug("X_test")
    EURUSDtdatalog.debug(X_test)
    EURUSDtdatalog.debug("Y_train")
    EURUSDtdatalog.debug(Y_train)
    EURUSDtdatalog.debug("Y_test")
    EURUSDtdatalog.debug(Y_test)
    model = keras.Sequential([
        layers.Dense(4, activation="relu", input_shape=[len(X[0])]),
        layers.Dense(12, activation="relu"),
        layers.Dense(36, activation="relu"),
        layers.Dense(108, activation="relu"),
        layers.Dense(324, activation="relu"),
        layers.Dense(648, activation="relu"),
        layers.Dense(324, activation="relu"),
        layers.Dense(108, activation="relu"),
        layers.Dense(36, activation="relu"),
        layers.Dense(12, activation="relu"),
        layers.Dense(4)
    ])
    model.compile(optimizer="adam", loss="mse")
    EURUSDmodellog.debug(f"{model.summary}")
    for layer in model.layers:
        weights, biases = layer.get_weights()
        EURUSDmodellog.debug(f"{layer.name}, {weights}, {biases}")
    model.fit(X_train, Y_train, epochs=traineraiEpochs, batch_size=traineraiBatchsize,
              validation_data=(X_test, Y_test))
    model.save(TaimodelS)
#**********************************************************************************************************************************
#**********************************************************************************************************************************
#**********************************************************************************************************************************    

# Continue running the script

# initialize the MT5 connection
if not mt5.initialize():
    print("Failed to initialize MT5 connection!")
    exit()

# Set the total runtime for the program
total_runtime = TotalRuntime 
converted_total_runtime = (total_runtime / 60) / 60

# Set the interval between script runs
script_interval = ScriptInterval

# Set the maximum number of allowed trades
max_trades = MaxTrades

# Get the start time for the program
start_time = time.time()

# Start the loop to run the script
while True:
    # Check if the total runtime has elapsed
    if time.time() - start_time >= total_runtime:
        print("Program has ended.")
        print(converted_total_runtime)
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
    script_path = "EURUSD/EURUSD_adata.py"
    subprocess.Popen(['python', script_path], creationflags=subprocess.CREATE_NO_WINDOW)

    # Wait for the interval
    time.sleep(script_interval)

# shutdown MT5 connection
mt5.shutdown()