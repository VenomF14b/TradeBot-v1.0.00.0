from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sklearn
import MetaTrader5 as mt5
import pyodbc
import alpaca_trade_api as tradeapi
import datetime as dt
import decimal
import time
import os
import tensorflow as tf
import subprocess





# Get a list of all files in the directory that start with "trained_model_"
model_files = [f for f in os.listdir('.') if f.startswith('trained_model_')]
# Sort the list by name (which will put the latest model at the end of the list)
model_files.sort()
# Load the last file in the list (which should be the latest model)
latest_model_file = model_files[-1]
model = load_model(latest_model_file)

# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')


# Load data from database
query = f"SELECT TOP 1 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00 ORDER BY timestamp DESC"
data = []
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
    data.append(row)
cursor.close()

# Convert data to numpy array and reverse the order of the rows
data = np.array(data[::-1])
X_new = data[:, 1:4]
O_data = data[:, 1:2] #open data

print("X_new")
print(X_new)
print("Odata")
print(O_data)

# Make a prediction on the new data point
Y_pred = model.predict(X_new)

# Extract the file name data of the model loaded and print on screen
model_name = latest_model_file.split(".")[0]
timestamp = model_name.split("_")[-1]
print(f"Loaded model from file: {latest_model_file}, created at {timestamp}")
print("X_new")
print(X_new)
print("Y_pred")
print("Predicted closing value:", Y_pred[0][0])


# Do something with the predictions
Open_adjust_up = 0.00003
if np.any(Y_pred > O_data + Open_adjust_up):
    print("Buy")
    print("Open Price Adjustor", O_data + Open_adjust_up)
    # Buy Code
    # Set up the API endpoint and credentials
    #endpoint = 'https://www.mql5.com/en/oauth/login'
    #key_id = 'q0bjra'
    #secret_key = 'briknghvpoqdwzdlqoqbnrdqxqsdcbzsqkfpjzxdqxfgbcpnrvjkmloknrmxodsn'

    # Initialize the API
    #api = tradeapi.REST(key_id, secret_key, endpoint, api_version='v2')
    #print(api)

    #account = api.get_account()

    #if account.status == 'ACTIVE':
    #    print('Authentication successful')
    #else:
    #    print('Authentication failed')

    # Get last trade for symbol
    #last_trade = api.get_last_trade(symbol=symbol)
    #latest_price = last_trade.price
    #print(f"Latest price: {latest_price}")

    # Calculate the quantity of shares to buy
    #cash_balance = float(api.get_account().cash)
    #qty = int(cash_balance / latest_price)

    # Place the buy order
    #api.submit_order(
    #symbol=symbol,
    #qty=qty,
    #side='buy',
    #type='market',
    #time_in_force='gtc'
    #)


Open_adjust_down = -0.00003
if np.any(Y_pred < O_data + Open_adjust_down):
   print("Sell")
   print("Open Price Adjustor", O_data + Open_adjust_down)
   # sell code here
else:
    print("Do nothing")
    # do nothing code here


# wait time before update
print("Please wait for the new set of data to import (default is set to 60 seconds)")
time.sleep(60)




# Connect to MT5
print("Establising Connection to MT5, Please wait")
mt5.initialize()
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1
print("Connection Successful")
print(symbol,"timeframe = " + str(timeframe))



# Calculate start and end times
end_time = dt.datetime.now()
start_time = end_time - dt.timedelta(days=60)
print("Data Time Start = " + str(start_time))
print("Data Time End = " + str(end_time))



# Get historical data
print("Getting historical data")
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
rates = np.array(rates)

# Update start and end times
start_time = end_time
end_time = dt.datetime.now()

# Establish a connection to the SQL Express database
print("Establishing a connection to the SQL Express database")
conn = pyodbc.connect('Driver={SQL Server};'
      'Server=VENOM-CLIENT\SQLEXPRESS;'
      'Database=TRADEBOT;'
      'Trusted_Connection=yes;')

# Write the data to the database
cursor = conn.cursor()
for rate in rates:
    timestamp = int(rate[0])

    # Check if timestamp already exists in the database
    cursor.execute("SELECT COUNT(*) FROM Tdata00 WHERE timestamp = ?", (timestamp,))
    count = cursor.fetchone()[0]
    if count == 0:

         # Write the data to the database
         values = [timestamp, float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7])]
         cursor.execute("INSERT INTO Tdata00 (timestamp, [open], high, low, [close], tick_volume, spread, real_volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
         print("database updated with the following data")
         print(values)
conn.commit()

# Load data from database
query = f"SELECT TOP 2 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00 ORDER BY timestamp DESC"
#query = "SELECT timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00 ORDER BY timestamp DESC"
data = []
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
    data.append(row)
cursor.close()

# Convert data to numpy array and reverse the order of the rows
data = np.array(data[::-1])
X = data[:, 1:4]  # open, high, low
Y = data[:, 4:5]  # close

print("X")
print(X)
print("Y")
print(Y)

# Normalize the data 
# Using MinMax normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)
# Using z-score normalization
#X = zscore(X)
#Y = zscore(Y)

print("X_Scaled")
print(X)
print("Y_Scaled")
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

# Check if a trained model exists and load it
try:
    model = tf.keras.models.load_model(latest_model_file)
    print("Loaded model from disk")
except:
    # Define the AI model if no trained model exists
    model = tf.keras.Sequential([
        layers.Dense(3, activation="relu", input_shape=[len(X[0])]),
        layers.Dense(8, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")
    print("Created new model")

# Train the model with new data
model.fit(X_train, Y_train, epochs=1000, batch_size=32,
          validation_data=(X_test, Y_test))

# Save the updated model
model.save(latest_model_file)


