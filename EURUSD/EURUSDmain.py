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


logging.basicConfig(filename='EURUSD/EURUSD.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
logging.info('\n''\nTraining Information')


# Ask user if they want to train a new model and overwrite old one
user_response = input("Do you want to train a new model and overwrite the old one? (y/n) ")

if user_response.lower() == "y":
    user_response = input("Do you want to change the training configuration? (y/n) ")
    if user_response.lower() == "y":
        #User input training parameters
        print("Define the training parameters")
        userIn_passedtime = input("Data request range in days:")
        if userIn_passedtime == "":
            userIn_passedtime = 7
            logging.debug(userIn_passedtime)
        try:
            userIn_passedtime = int(userIn_passedtime)
        except ValueError:
            print("Invalid input! Please enter an integer value.")

        userIn_traineraiRowselector = input("Training data size number of rows: ")
        if userIn_traineraiRowselector == "":
            userIn_traineraiRowselector = 10080
            logging.debug(userIn_traineraiRowselector)
        try:
            userIn_traineraiRowselector = int(userIn_traineraiRowselector)
        except ValueError:
            print("Invalid input! Please enter an integer value.")

        userIn_traineraiEpochs = input("Training Epochs size: ")
        if userIn_traineraiEpochs == "":
            userIn_traineraiEpochs = 100
            logging.debug(userIn_traineraiEpochs)
        try:
            userIn_traineraiEpochs = int(userIn_traineraiEpochs)
        except ValueError:
            print("Invalid input! Please enter an integer value.")

        userIn_traineraiBatchsize = input("Training batch size: ")
        if userIn_traineraiBatchsize == "":
            userIn_traineraiBatchsize = 5
            logging.debug(userIn_traineraiBatchsize)
        try:
            userIn_traineraiBatchsize = int(userIn_traineraiBatchsize)
        except ValueError:
            print("Invalid input! Please enter an integer value.")

        # Create a ConfigParser object
        config = configparser.ConfigParser()
            
        # Set the values for the configuration file
        config['Training Parameters'] = {
            'passedtime': int(userIn_passedtime),
            'traineraiRowselector': int(userIn_traineraiRowselector),
            'traineraiEpochs': int(userIn_traineraiEpochs),
            'traineraiBatchsize': int(userIn_traineraiBatchsize)
        }

        # Write the configuration to a file
        with open('EURUSD/configEURUSD.ini', 'w') as configfile:
            config.write(configfile)


    # Read the configuration to a file
    config = configparser.ConfigParser()
    config.read('EURUSD/configEURUSD.ini')
    training_params = config['Training Parameters']

    #Update adata
    symbol = "EURUSD" #Symbol selector
    passedtime = days= + int(training_params.get('passedtime', '7')) #Historical data time adjustor in days
    #Trainerai
    traineraiRowselector = int(training_params.get('traineraiRowselector', '10080'))
    traineraiEpochs = int(training_params.get('traineraiEpochs', '100'))
    traineraiBatchsize = int(training_params.get('traineraiBatchsize', '5'))
    TaimodelS = "EURUSD/EURUSD.h5" #Model to save
    # Run the script to train a new model
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    symbol = symbol
    timeframe = mt5.TIMEFRAME_M1
    #cmd info
    print("Training the new model") # Training the new model
    logging.debug("Connection to MetaTrader5 successful")
    #logging info
    logging.debug("Training the new model") # Training the new model
    logging.debug("Connection to MetaTrader5 successful")

    end_time = dt.datetime.now()    # Calculate start and end times
    end_time += dt.timedelta(hours=3)
    start_time = end_time - dt.timedelta(passedtime)   
    #cmd info
    logging.debug("Data Time Start = " + str(start_time))
    logging.debug("Data Time End = " + str(end_time))

    logging.debug("Getting historical data")    # Get historical data
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
             logging.debug(values)
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
    logging.debug("X")
    logging.debug(X)
    logging.debug("Y")
    logging.debug(Y)

    # Normalize the data
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #X = scaler.fit_transform(X)
    #Y = scaler.fit_transform(Y)

    #logging.debug("X Normalized")
    #logging.debug(X)
    #logging.debug("Y Normalized")
    #logging.debug(Y)

    # Split the data into training and testing sets
    split = int(0.70 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    logging.debug("X_train")
    logging.debug(X_train)
    logging.debug("X_test")
    logging.debug(X_test)
    logging.debug("Y_train")
    logging.debug(Y_train)
    logging.debug("Y_test")
    logging.debug(Y_test)

    # Define the AI model
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

    # Print the model summary
    logging.debug(f"{model.summary}")

    # Print the model weights and biases
    for layer in model.layers:
        weights, biases = layer.get_weights()
        logging.debug(f"{layer.name}, {weights}, {biases}")

    # Train the model
    model.fit(X_train, Y_train, epochs=traineraiEpochs, batch_size=traineraiBatchsize,
              validation_data=(X_test, Y_test))

    # Save the model
    model.save(TaimodelS)
    

# Continue running the script



# Ask user if they want to change the running models parameters
user_response = input("Do you want to change the running models parameters? (y/n) ")

if user_response.lower() == "y":
    #User input constantai parameters
    print("Define the running parameters")
    userIn_passedtimeConstantai = input("Running rates data request range in days:")
    if userIn_passedtimeConstantai == "":
        userIn_passedtimeConstantai = 1
        logging.debug(userIn_passedtimeConstantai)
    try:
        userIn_passedtimeConstantai = int(userIn_passedtimeConstantai)
    except ValueError:
        print("Invalid input! Please enter an integer value.")

    userIn_passedtimeWLdataConstantai = input("Running win loss data request range in days:")
    if userIn_passedtimeWLdataConstantai == "":
        userIn_passedtimeWLdataConstantai = 7
        logging.debug(userIn_passedtimeWLdataConstantai)
    try:
        userIn_passedtimeWLdataConstantai = int(userIn_passedtimeWLdataConstantai)
    except ValueError:
        print("Invalid input! Please enter an integer value.")

    userIn_constantaiRowselector = input("Running reinforced training data size number of rows @ 60 second intervals:")
    if userIn_constantaiRowselector == "":
        userIn_constantaiRowselector = 7
        logging.debug(userIn_constantaiRowselector)
    try:
        userIn_constantaiRowselector = int(userIn_constantaiRowselector)
    except ValueError:
        print("Invalid input! Please enter an integer value.")

    userIn_wldataRowselector = input("Running reinforced Win loss training data size number of rows:")
    if userIn_wldataRowselector == "":
        userIn_wldataRowselector = 8
        logging.debug(userIn_wldataRowselector)
    try:
        userIn_wldataRowselector = int(userIn_wldataRowselector)
    except ValueError:
        print("Invalid input! Please enter an integer value.")

    userIn_constantaiTrainsplit = input("Running Training data split example '0.70' 70/30 training/testing:")
    if userIn_constantaiTrainsplit == "":
        userIn_constantaiTrainsplit = 0.70
        logging.debug(userIn_constantaiTrainsplit)
    try:
        userIn_constantaiTrainsplit = float(userIn_constantaiTrainsplit)
    except ValueError:
        print("Invalid input! Please enter a float value.")

    userIn_constantaiEpochs = input("Runnung training Epochs size: ")
    if userIn_constantaiEpochs == "":
        userIn_constantaiEpochs = 5
        logging.debug(userIn_constantaiEpochs)
    try:
        userIn_constantaiEpochs = int(userIn_constantaiEpochs)
    except ValueError:
        print("Invalid input! Please enter an integer value.")

    userIn_constantaiBatchsize = input("Runnung training Batch size: ")
    if userIn_constantaiBatchsize == "":
        userIn_constantaiBatchsize = 1
        logging.debug(userIn_constantaiBatchsize)
    try:
        userIn_constantaiBatchsize = int(userIn_constantaiBatchsize)
    except ValueError:
        print("Invalid input! Please enter an integer value.")

    userIn_buyVolume = input("Set buy volume example 0.01: ")
    if userIn_buyVolume == "":
        userIn_buyVolume = 0.01
        logging.debug(userIn_buyVolume)
    try:
        userIn_buyVolume = float(userIn_buyVolume)
    except ValueError:
        print("Invalid input! Please enter an integer value.")

    userIn_buyStoploss = input("Set buy StopLoss example 0.0001: ")
    if userIn_buyStoploss == "":
        userIn_buyStoploss = 0.0001
        logging.debug(userIn_buyStoploss)
    try:
        userIn_buyStoploss = float(userIn_buyStoploss)
    except ValueError:
        print("Invalid input! Please enter an integer value.")

    userIn_buyTakeProfit = input("Set buy TakeProfit example 0.0001: ")
    if userIn_buyTakeProfit == "":
        userIn_buyTakeProfit = 0.0001
        logging.debug(userIn_buyTakeProfit)
    try:
        userIn_buyTakeProfit = float(userIn_buyTakeProfit)
    except ValueError:
        print("Invalid input! Please enter an integer value.")

    userIn_buyMagic = input("Set 6 digit buy identifier for the buy action: ")
    if userIn_buyMagic == "":
        userIn_buyMagic = 123456
        logging.debug(userIn_buyMagic)
    try:
        userIn_buyMagic = int(userIn_buyMagic)
    except ValueError:
        print("Invalid input! Please enter an integer value example 123456.")

    userIn_sellVolume = input("Set Sell volume example 0.01: ")
    if userIn_sellVolume == "":
        userIn_sellVolume = 0.01
        logging.debug(userIn_sellVolume)
    try:
        userIn_sellVolume = float(userIn_sellVolume)
    except ValueError:
        print("Invalid input! Please enter a float value.")

    userIn_sellStoploss = input("Set Sell StopLoss example 0.0001: ")
    if userIn_sellStoploss == "":
        userIn_sellStoploss = 0.0001
        logging.debug(userIn_sellStoploss)
    try:
        userIn_sellStoploss = float(userIn_sellStoploss)
    except ValueError:
        print("Invalid input! Please enter a float value.")

    userIn_sellTakeProfit = input("Set Sell TakeProfit example 0.0001: ")
    if userIn_sellTakeProfit == "":
        userIn_sellTakeProfit = 0.0001
        logging.debug(userIn_sellTakeProfit)
    try:
        userIn_sellTakeProfit = float(userIn_sellTakeProfit)
    except ValueError:
        print("Invalid input! Please enter a float value.")

    userIn_sellMagic = input("Set 6 digit Sell identifier for the buy action: ")
    if userIn_sellMagic == "":
        userIn_sellMagic = 123456
        logging.debug(userIn_sellMagic)
    try:
        userIn_sellMagic = int(userIn_sellMagic)
    except ValueError:
        print("Invalid input! Please enter an integer value example 123456.")


    # Create a ConfigParser object
    config = configparser.ConfigParser()
            
    # Set the values for the configuration file
    config['Constantai Parameters'] = {
        'passedtimeConstantai': int(userIn_passedtimeConstantai),
        'passedtimeWLdataConstantai': int(userIn_passedtimeWLdataConstantai),
        'constantaiRowselector': int(userIn_constantaiRowselector),
        'wldataRowselector': int(userIn_wldataRowselector),
        'constantaiTrainsplit': float(userIn_constantaiTrainsplit),
        'constantaiEpochs': int(userIn_constantaiEpochs),
        'constantaiBatchsize': int(userIn_constantaiBatchsize),
        'buyVolume': float(userIn_buyVolume),
        'buyStoploss': float(userIn_buyStoploss),
        'buyTakeProfit': float(userIn_buyTakeProfit),
        'buyMagic': int(userIn_buyMagic),
        'sellVolume': float(userIn_sellVolume),
        'sellStoploss': float(userIn_sellStoploss),
        'sellTakeProfit': float(userIn_sellTakeProfit),
        'sellMagic': int(userIn_sellMagic)
        }

    # Write the configuration to a file
    with open('EURUSD/configEURUSDa.ini', 'w') as configfile:
        config.write(configfile)

time.sleep(1)

# initialize the MT5 connection
if not mt5.initialize():
    print("Failed to initialize MT5 connection!")
    exit()

# Set the total runtime for the program
total_runtime = 21600  # seconds
converted_total_runtime = (total_runtime / 60) / 60

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
    subprocess.Popen(['python', script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)

#    # Run the script minimized
#    script_path = "EURUSD/EURUSD_adata.py"
#    creation_flags = subprocess.CREATE_NEW_CONSOLE | win32con.SW_MINIMIZE
#    subprocess.Popen(['python', script_path], creationflags=creation_flags)

    # Wait for the interval
    time.sleep(script_interval)

# shutdown MT5 connection
mt5.shutdown()