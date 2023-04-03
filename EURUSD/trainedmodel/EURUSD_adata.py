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
wldataTimeframe = days=4 #Win Loss data time adjustor in days
#constantai
constantaiRowselector = 3 #Number of rows to load from adata in decending order
wldataRowselector = 4 #Number of rows to load from wldata in decending order
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
#Logging
logging.debug("Establising Connection to MT5, Please wait")
logging.debug("Connection Successful")
logging.debug("%s timeframe = %d", symbol, timeframe)
print("Establising Connection to MT5, Please wait")

# Calculate start and end times
end_time = dt.datetime.now()
end_time += dt.timedelta(hours=3)
start_time = end_time - dt.timedelta(passedtime)
#Logging
logging.debug("Data Time Start = " + str(start_time))
logging.debug("Data Time End = " + str(end_time))
print("Data Time Start = " + str(start_time))
print("Data Time End = " + str(end_time))
print("Please wait")

# Get historical data
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
rates = np.array(rates)
#Logging
logging.debug("Pulled historical data")
print("Pulled historical data")

# Establish a connection to the SQL Express database
logging.debug("Establishing a connection to the SQL Express database")
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')
#Logging
logging.debug("Connection established")
print("Connection established")

cursor = conn.cursor()
for rate in rates:
    timestamp = int(rate[0])

    # Check if timestamp already exists in the database
    cursor.execute("SELECT COUNT(*) FROM EURUSDAdata WHERE timestamp = ?", (timestamp,))
    #cursor.execute("SELECT COUNT(*) FROM (SELECT TOP 500 * FROM EURUSDAdata ORDER BY timestamp DESC) AS latest WHERE timestamp = ?", (timestamp,))
    count = cursor.fetchone()[0]
    if count == 0:

         # Write the data to the database
         values = [timestamp, float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7])]
         cursor.execute("INSERT INTO EURUSDAdata (timestamp, [open], high, low, [close], tick_volume, spread, real_volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
         #Logging
         logging.debug("Latest Timeframe data")
         logging.debug(values)
         print("Latest Timeframe data")
         print(values)

cursor.commit()
#Logging
logging.debug("SQL complete MT data is up to date")
print("SQL complete MT data is up to date")

#**********************************************************************************************************************************
#Gets the WLdata and writes to db *************************************************************************************************
#**********************************************************************************************************************************
print("WLdata get code")
print("Updating WL data")

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

print(to_date)
print(from_date)

# get deals for symbols whose names contain "EURUSD" within a specified interval
#try code_________
rates = mt5.history_deals_get(from_date, to_date, group="*EURUSD*")
rates = [deal for deal in deals if deal.profit != 0]
rates = np.array(deals)
#Logging
logging.debug("Pulled historical data")
print("Pulled historical data")
print(rates)

#deals=mt5.history_deals_get(from_date, to_date, group="*EURUSD*")
# filter deals with zero profit
#deals = [deal for deal in deals if deal.profit != 0]
# sort the dataframe by ticket in ascending order
#print(deals)
#deals = sorted(deals, key=lambda deal: deal.ticket)

#edit from here trying to use rates rather than deals 

if deals==None:
    print("No deals with group=\"*EURUSD*\", error code={}".format(mt5.last_error()))
elif len(deals)> 0:
    print("history_deals_get({}, {}, group=\"*EURUSD*\")={}".format(from_date,to_date,len(deals)))


    # display these deals as a table using pandas.DataFrame
    df=pd.DataFrame(list(deals),columns=deals[0]._asdict().keys())
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['time'] = df['time'].apply(lambda x: int(x.timestamp()))
    print(df)
#print("")

cursor = conn.cursor()

# create table to store deals data
#cursor.execute("CREATE TABLE EURUSDWLdata (ticket INT, [order] INT, time VARCHAR(255), type INT, entry FLOAT, magic INT, position_ID INT, reason INT, volume FLOAT, price FLOAT, commission FLOAT, swap FLOAT, profit FLOAT, fee FLOAT, symbol VARCHAR(50), comment VARCHAR(255), external_ID VARCHAR(255))")

# insert deals data into the table
for deal in deals:
    # execute a SELECT query to check if the position ID exists in the table
    cursor.execute(f"SELECT * FROM EURUSDAdata WHERE timestamp <= {deal.time + 60} ORDER BY timestamp DESC")
    result = cursor.fetchone()
    
    if result is not None:
        # update the existing row with the new data
        cursor.execute(f"UPDATE EURUSDAdata SET ticket=?, [order]=?, time=?, type=?, entry=?, magic=?, position_id=?, reason=?, volume=?, price=?, commission=?, swap=?, profit=?, fee=?, symbol=?, comment=?, external_id=? WHERE timestamp=?", 
                       (deal.ticket, deal.order, deal.time, deal.type, deal.entry, deal.magic, deal.position_id, deal.reason, deal.volume, deal.price, deal.commission, deal.swap, deal.profit, deal.fee, deal.symbol, deal.comment, deal.external_id, result[0]))
        print("Deal data updated")
        # commit changes
        conn.commit()
        
    else:
        # do nothing if there is no match
        print("No matching data found")
        
    # check for null values in the profit column
    cursor.execute("SELECT * FROM EURUSDAdata WHERE profit is NULL")
    null_profit_rows = cursor.fetchall()
    
    if null_profit_rows:
        # update rows with null profit values to 0
        for row in null_profit_rows:
            cursor.execute(f"UPDATE EURUSDAdata SET profit=0 WHERE timestamp={row[0]}")
        print("Profit data updated")
    # commit changes and close connection
    conn.commit()
conn.close()

#***********************************************************************************************************************************
#Constant ai runs itterations to tweak weights**************************************************************************************
#***********************************************************************************************************************************
print("Adjusting weights with new data")

# Connect to the SQL Express database
server = 'VENOM-CLIENT\SQLEXPRESS'
database = 'NSAI'
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')

# Load adata from database
query = f"SELECT TOP ({constantaiRowselector}) timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
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
#Logging
logging.debug("X")
logging.debug(X)
logging.debug("Y")
logging.debug(Y)

# Load wldata from database
query = f"SELECT TOP ({wldataRowselector}) position_ID, profit FROM EURUSDAdata ORDER BY position_ID DESC"
profit_data = []
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
    profit_data.append(row)

print(profit_data)

# Convert data to numpy array and reverse the order of the rows
if len(profit_data) == 0:
    profit_data = np.array([[0, 0]])
else:
    profit_data = np.array(profit_data[::-1])
profit = profit_data[:-1, 1]  # Select only the second column

# Convert data to numpy array and reverse the order of the rows
#profit_data = np.array(profit_data[::-1])
print(profit_data)
#profit = profit_data[:-1, 1]  # Select only the second column
print(profit)

cursor.close()
#Logging
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
#Logging
logging.info("Reward")
logging.debug(reward)

# Split the data into training and testing sets
split = int((constantaiTrainsplit) * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
R_train, R_test = reward[:split], reward[split:]

#Logging
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
print(Adata_Actual)

#Logging
logging.debug("Latest data")
logging.debug(X_new)

next_timestamp = data[-1, 0] + 60

#Logging
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

#Logging
logging.debug(X_latest)

# Make a prediction on the new data point
Y_pred = model.predict(X_latest)
#Logging
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
print(next_timestamp)
print(latest_timestamp)
print(next_RowTimestamp)

# Define the SQL statement to insert the row with the predicted values
# Write the data to the database
values = [int(next_timestamp), float(Pred_Open), float(Pred_High), float(Pred_Low), float(Pred_Close)]
cursor.execute("INSERT INTO EURUSDPdata (timestamp, pred_open, pred_high, pred_low, pred_close) VALUES (?, ?, ?, ?, ?)", tuple(values))
print(values)

# Commit the changes to the database
conn.commit()

# Close the database connection
conn.close()





# Do something with the predictions
Decision_Adjustor_Buy = buyadataAdjustor
Last_Close_Buy_Helper = Last_Close + Decision_Adjustor_Buy
if np.any(Pred_Close > Last_Close_Buy_Helper):
    #Logging
    logging.debug("Last_Close_Buy_Helper")
    logging.debug(Last_Close_Buy_Helper)
    logging.debug("Buy")   
    # Buy Code

    # connect to MetaTrader 5
    if not mt5.initialize():
        #Logging
        logging.error("initialize() failed")
        mt5.shutdown()

    # define the symbol and order type
    symbol = symbol
    lot_size = buyVolume
    stop_loss = buyStoploss
    take_profit = buyTakeProfit
    magic_number = buyMagic
    price = mt5.symbol_info_tick(symbol).ask
    type = mt5.ORDER_TYPE_BUY

    # Do something with the price
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
        #Logging
        logging.error("order failed with retcode={}".format(result.retcode))
        logging.error("message={}".format(result.comment))
        print("BUY ORDER FAILED")
        print("order failed with retcode={}".format(result.retcode))
        print("message={}".format(result.comment))
    else:
        #Logging
        logging.debug("order executed with order_id={}".format(result.order))
        logging.debug("BUY")
        print("BUYING")
else:
    Decision_Adjustor_Sell = selladataAdjustor
    Last_Close_Sell_Helper = Last_Close + Decision_Adjustor_Sell
    if np.any(Pred_Close < Last_Close_Sell_Helper):
        #Logging
       logging.debug("Last_Close_Sell_Helper")
       logging.debug(Last_Close_Sell_Helper)
       logging.debug("Sell")
       # sell code here
       # connect to MetaTrader 5
       if not mt5.initialize():
           logging.debug("initialize() failed")
           mt5.shutdown()

       # define the symbol and order type
       symbol = symbol
       lot_size = sellVolume
       stop_loss = sellStoploss
       take_profit = sellTakeProfit
       magic_number = sellMagic
       price = mt5.symbol_info_tick(symbol).bid
       type = mt5.ORDER_TYPE_SELL

       # Do something with the price
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
           #Logging
           logging.debug("order failed with retcode={}".format(result.retcode))
           logging.debug("message={}".format(result.comment))
           print("SELL ORDER FAILED")
           print("order failed with retcode={}".format(result.retcode))
           print("message={}".format(result.comment))
       else:
           #Logging
           logging.debug("order executed with order_id={}".format(result.order))
           logging.debug("SELL")
           print("SELLING")

    else:
        #Logging
        logging.debug("Do nothing")
        print("DO NOTHING!!!")
        # do nothing code here
        mt5.shutdown
time.sleep(10)
