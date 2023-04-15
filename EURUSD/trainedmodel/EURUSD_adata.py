import MetaTrader5 as mt5
import numpy as np
import tensorflow as tf
import pandas as pd
import datetime as dt
import tkinter as tk
import alpaca_trade_api as tradeapi
import datetime
import sklearn
import glob
import pyodbc
import decimal
import subprocess
import time
import ctypes
import signal
import sys
import os
import logging
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, timezone
from tkinter import messagebox
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore



#adata
adataTimeframe = mt5.TIMEFRAME_M1 #Timeframe selector
symbol = "EURUSD" #Symbol selector
passedtime = days=7 #Historical data time adjustor in days
#wldata
wldataTimeframe = days=7 #Win Loss data time adjustor in days
#constantai
constantaiRowselector = 5 #Number of rows to load from adata in decending order
wldataRowselector = 25 #Number of rows to load from wldata in decending order
constantaiTrainsplit = 0.70 #Training and testing data split
constantaiEpochs = 5
constantaiBatchsize = 1
Tmodel = "EURUSD/EURUSD.h5" #Model to load
TmodelS = "EURUSD/EURUSD.h5" #Model to save
#Predict
next_RowTimestamp = 60 # Calculate the timestamp for the next row this value is in unix time
#Buying
buyadataAdjustor = 0.000000 #Adjusts to adata latest data adjust in positive range
buyVolume = 0.01 #The volume of trades to buy
buyStoploss = 0.0001 #Stop loss for buy action
buyTakeProfit = 0.0001 #Take profit for buy action
buyMagic = 123456 # can identify
#Selling
selladataAdjustor = -0.000000 #Adjusts to adata latest data adjust in negative range
sellVolume = 0.01 #The volume of trades to sell
sellStoploss = 0.0001 #Stop loss for sell action
sellTakeProfit = 0.0001 #Take profit for sell action
sellMagic = 123456 # can identify


# Continue running the script
logging.basicConfig(filename='EURUSD.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
logging.info('\n''\nUpdating Information')

#**********************************************************************************************************************************
#Gets the histdata and writes to db *************************************************************************************************
#**********************************************************************************************************************************

# Connect to MT5
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
symbol = symbol
timeframe = adataTimeframe
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Connection Successful")
logging.debug("%s timeframe = %d", symbol, timeframe)

# Calculate start and end times
end_time = dt.datetime.now()
end_time += dt.timedelta(hours=3)
start_time = end_time - dt.timedelta(passedtime)
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Data Time Start = " + str(start_time))
logging.debug("Data Time End = " + str(end_time))

# Get historical data
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
rates = np.array(rates)
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Rates data raw MT5")
logging.debug(rates)

# Establish a connection to the SQL Express database
logging.debug("Establishing a connection to the SQL Express database")
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()
for rate in rates:
    timestamp = int(rate[0])

    # Check if timestamp already exists in the database
    cursor.execute("SELECT COUNT(*) FROM EURUSDAdata WHERE timestamp = ?", (timestamp,))
    count = cursor.fetchone()[0]
    if count == 0:
         # Write the data to the database
         values = [timestamp, float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7])]
         cursor.execute("INSERT INTO EURUSDAdata (timestamp, [open], high, low, [close], tick_volume, spread, real_volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
         #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
         logging.debug("Latest written Timeframe data")
         logging.debug(values)
         logging.debug("Rates Updated")
cursor.commit()


#**********************************************************************************************************************************
#Gets the WLdata and writes to db *************************************************************************************************
#**********************************************************************************************************************************
pd.set_option('display.max_columns', 1000) # number of columns to be displayed
pd.set_option('display.width', 3000)      # max table width to display

# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)
print()

# establish connection to the MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Calculate start and end times
to_date = dt.datetime.now() + dt.timedelta(hours=3)
from_date = to_date - dt.timedelta(wldataTimeframe)
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.info("To date")
logging.info(to_date)
logging.info("From date:")
logging.info(from_date)

# get deals for symbols whose names contain "EURUSD" within a specified interval
deals=mt5.history_deals_get(from_date, to_date, group="*EURUSD*")
# filter deals with zero profit
deals = [deal for deal in deals if deal.profit != 0]

if deals==None:
    print("No deals with group=\"*EURUSD*\", error code={}".format(mt5.last_error()))
elif len(deals)> 0:
    print("history_deals_get({}, {}, group=\"*EURUSD*\")={}".format(from_date,to_date,len(deals)))

cursor = conn.cursor()

# create table to store deals data
#cursor.execute("CREATE TABLE EURUSDWLdata (ticket INT, [order] INT, time VARCHAR(255), type INT, entry FLOAT, magic INT, position_ID INT, reason INT, volume FLOAT, price FLOAT, commission FLOAT, swap FLOAT, profit FLOAT, fee FLOAT, symbol VARCHAR(50), comment VARCHAR(255), external_ID VARCHAR(255))")

# insert deals data into the table

for deal in deals:
    # execute a SELECT query to check if the position ID exists in the table
    cursor.execute(f"SELECT time FROM EURUSDWLdata WHERE time = {deal.time}")
    result = cursor.fetchone()
    if result is None:
        # define a tuple with the values to insert
        values = (deal.ticket, deal.order, deal.time, deal.type, deal.entry, deal.magic, deal.position_id, deal.reason, deal.volume, deal.price, deal.commission, deal.swap, deal.profit, deal.fee, deal.symbol, deal.comment, deal.external_id)
        # execute the INSERT statement with the tuple
        cursor.execute("INSERT INTO EURUSDWLdata (ticket, [order], time, type, entry, magic, position_id, reason, volume, price, commission, swap, profit, fee, symbol, comment, external_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.debug(values)
logging.debug("Deal data updated")
conn.commit()
conn.close()

#***********************************************************************************************************************************
#Constant ai runs itterations to tweak weights**************************************************************************************
#***********************************************************************************************************************************
print("Adjusting model")

# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Load data from database
query = f"SELECT TOP ({constantaiRowselector}) timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
cursor = conn.cursor()
cursor.execute(query)
data = cursor.fetchall()
for row in cursor:
    data.append(list(row))
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Rates data raw")
logging.debug(data)

# Load profit data from EURUSDWLdata
query = f"SELECT TOP ({wldataRowselector}) time, profit FROM EURUSDWLdata ORDER BY time DESC"
cursor = conn.cursor()
cursor.execute(query)
profit_data = cursor.fetchall()
for row in cursor:
    profit_data.append(list(row))
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Profit data raw")
logging.debug(profit_data)
cursor.close()

# Match profit data to timestamps in EURUSDAdata
i = 0
for j, row in enumerate(data):
    timestamp = row[0]
    next_timestamp = data[j+1][0] if j < len(data)-1 else timestamp + 60
    if i < len(profit_data) and profit_data[i][0] >= timestamp and profit_data[i][0] < next_timestamp:
        profit = profit_data[i][1]
        i += 1
    else:
        profit = 0
    row_tuple = tuple(row) + (profit,)
    data[j] = list(row_tuple)
    data[j].append(profit)
    #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    logging.debug("data[j]")
    logging.debug(data[j])

# Convert data to numpy array and reverse the order of the rows
data = np.array(data[::-1])
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
X = data[:, 1:5]  # timestamp, [open], high, low, [close]
Y = np.roll(data[:, 1:5], -1, axis=0) # Shift the Y values by one time step to predict the next set of datapoints
profit = data[:, 5:6]
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Input data <X>")
logging.debug(X)
logging.debug("Output data <Y>")
logging.debug(Y)
logging.debug("Profit data <profit>")
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
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.info("Reward from profit data in np array")
logging.debug(reward)

# Split the data into training and testing sets
split = int((constantaiTrainsplit) * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
R_train, R_test = reward[:split], reward[split:]
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Input training data <X_train>")
logging.debug(X_train)
logging.debug("Input testing data <X_test>")
logging.debug(X_test)
logging.debug("Output training data <Y_train>")
logging.debug(Y_train)
logging.debug("Output testing data <Y_test>")
logging.debug(Y_test)
logging.debug("Reward training data <R_train>")
logging.debug(R_train)
logging.debug("Reward testing data <R_test>")
logging.debug(R_test)

model = load_model(Tmodel)

# Train the model
model.fit(X_train, Y_train, epochs=constantaiEpochs, batch_size=constantaiBatchsize, sample_weight=R_train,
          validation_data=(X_test, Y_test), validation_steps=len(X_test),
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)])

model.save(TmodelS)


#Predict and buy sell conditions****************************************************************************
print("Predicting with trained data and new data")

model = load_model(Tmodel)

# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Load adata from database
query = f"SELECT TOP 1 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
data = []
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
    data.append(row)
cursor.close()

# Convert data to numpy array and reverse the order of the rows
data = np.array(data[::-1])
X_new = data[:, 1:5]
O_data = data[:, 1:5] 
Adata_Actual = data[: 1:5]
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Latest Actual rates data")
logging.debug(Adata_Actual)
logging.debug("Latest rates data")
logging.debug(X_new)

next_timestamp = data[-1, 0] + 60

#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("PRediction time stamp")
logging.debug(next_timestamp)

# Create new input data X_new
X_latest = np.array([[
    next_timestamp,  # Timestamp for the next time frame
    data[-1, 1],             # Placeholder value for open
    data[-1, 2],             # Placeholder value for high
    data[-1, 3],             # Placeholder value for low
    data[-1, 4]              # Placeholder value for close
]])

X_latest = X_latest[:, 1:] # Select all rows and columns 1 to 4 (inclusive

#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Data before the prediction")
logging.debug(X_latest)

# Make a prediction on the new data point
Y_pred = model.predict(X_latest)
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Prediction: %s", str(Y_pred[0]))
Pred_Open = Y_pred[0,0]
Pred_High = Y_pred[0,1]
Pred_Low = Y_pred[0,2]
Pred_Close = Y_pred[0,3]
Last_Open = O_data[0,0]
Last_High = O_data[0,1]
Last_Low = O_data[0,2]
Last_Close = O_data[0,3]

# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Create a cursor object
cursor = conn.cursor()

# Get the latest timestamp from the database
query_latest_timestamp = "SELECT TOP 1 timestamp FROM EURUSDAdata ORDER BY timestamp DESC"
cursor.execute(query_latest_timestamp)
latest_timestamp = cursor.fetchone()[0]



# Calculate the timestamp for the next row
next_timestamp = latest_timestamp + next_RowTimestamp
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Prediction data timestamp calculation")
logging.debug("Next timestamp")
logging.debug(next_timestamp)
logging.debug("Latest timestamp")
logging.debug(latest_timestamp)
logging.debug("Next row timestamp")
logging.debug(next_RowTimestamp)

# Define the SQL statement to insert the row with the predicted values
# Write the data to the database
values = [int(next_timestamp), float(Pred_Open), float(Pred_High), float(Pred_Low), float(Pred_Close)]
cursor.execute("INSERT INTO EURUSDPdata (timestamp, pred_open, pred_high, pred_low, pred_close) VALUES (?, ?, ?, ?, ?)", tuple(values))
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Prediction values witten to EURUSDPdata")
logging.debug(values)

conn.commit()
conn.close()

# Do something with the predictions
Decision_Adjustor_Buy = buyadataAdjustor
Last_Close_Buy_Helper = Last_Close + Decision_Adjustor_Buy
if np.any(Pred_Close > Last_Close_Buy_Helper):
    #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    logging.debug("Last_Close_Buy_Helper")
    logging.debug(Last_Close_Buy_Helper)
    # Buy Code

    # connect to MetaTrader 5
    if not mt5.initialize():
        mt5.shutdown()
        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.error("initialize() failed")

    # define the symbol and order type
    symbol = symbol
    lot_size = buyVolume
    stop_loss = buyStoploss
    take_profit = buyTakeProfit
    magic_number = buyMagic
    price = mt5.symbol_info_tick(symbol).ask
    type = mt5.ORDER_TYPE_BUY

    # Do something with the price
    #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    logging.debug(f"The latest price for {symbol} is {price}.")
        # create a request for a new order
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": type,
        "price": price,
        "sl": price - stop_loss,
        "tp": price + take_profit,
        "magic": magic_number,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # send the order request
    #result = mt5.orders_send(request)
    result = mt5.order_send(request)

    # check if the order was executed successfully
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.error("order failed with retcode={}".format(result.retcode))
        logging.error("message={}".format(result.comment))
        print("BUY ORDER FAILED")
        print("order failed with retcode={}".format(result.retcode))
        print("message={}".format(result.comment))
    else:
        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.debug("order executed with order_id={}".format(result.order))
        logging.debug("BUY")
        print("BUYING")
else:
    Decision_Adjustor_Sell = selladataAdjustor
    Last_Close_Sell_Helper = Last_Close + Decision_Adjustor_Sell
    if np.any(Pred_Close < Last_Close_Sell_Helper):
        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
       logging.debug("Last_Close_Sell_Helper")
       logging.debug(Last_Close_Sell_Helper)
       # sell code here
       # connect to MetaTrader 5
       if not mt5.initialize():
           mt5.shutdown()
           #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
           logging.debug("initialize() failed")

       # define the symbol and order type
       symbol = symbol
       lot_size = sellVolume
       stop_loss = sellStoploss
       take_profit = sellTakeProfit
       magic_number = sellMagic
       price = mt5.symbol_info_tick(symbol).bid
       type = mt5.ORDER_TYPE_SELL

       # Do something with the price
       #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
       logging.debug(f"The latest price for {symbol} is {price}.")
       # create a request for a new order
       request = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbol,
           "volume": lot_size,
           "type": type,
           "price": price,
           "sl": price + stop_loss,
           "tp": price - take_profit,
           "magic": magic_number,
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
       }

       # send the order request
       result = mt5.order_send(request)
        # check if the order was executed successfully
       if result.retcode != mt5.TRADE_RETCODE_DONE:
           #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
           logging.debug("order failed with retcode={}".format(result.retcode))
           logging.debug("message={}".format(result.comment))
           print("SELL ORDER FAILED")
           print("order failed with retcode={}".format(result.retcode))
           print("message={}".format(result.comment))
       else:
           #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
           logging.debug("order executed with order_id={}".format(result.order))
           logging.debug("SELL")
           print("SELLING")

    else:
        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.debug("Do nothing")
        print("DO NOTHING!!!")
        # do nothing code here
        mt5.shutdown
time.sleep(10)
