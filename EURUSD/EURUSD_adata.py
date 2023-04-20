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
import configparser
import csv
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, timezone
from tkinter import messagebox
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore



# Read the configuration to a file
config = configparser.ConfigParser()
config.read('EURUSD/configEURUSDa.ini')
Constantai_params = config['Constantai Parameters']

#adata
adataTimeframe = mt5.TIMEFRAME_M1 #Timeframe selector
symbol = "EURUSD" #Symbol selector
passedtimeConstantai = days= + int(Constantai_params.get('passedtimeConstantai', '1')) #Historical data time adjustor in days
#wldata
wldataTimeframe = days= + int(Constantai_params.get('passedtimeWLdataConstantai', '7')) #Win Loss data time adjustor in days
#constantai
constantaiRowselector = int(Constantai_params.get('constantaiRowselector','3')) #Number of rows to load from adata in decending order
wldataRowselector = int(Constantai_params.get('wldataRowselector', '4')) #Number of rows to load from wldata in decending order
constantaiTrainsplit = float(Constantai_params.get('constantaiTrainsplit','0.70')) #Training and testing data split
constantaiEpochs = int(Constantai_params.get('constantaiEpochs', '5'))
constantaiBatchsize = int(Constantai_params.get('constantaiBatchsize', '1'))
Tmodel = "EURUSD/EURUSD.h5" #Model to load
TmodelS = "EURUSD/EURUSD.h5" #Model to save
#Predict
next_RowTimestamp = 60 # Calculate the timestamp for the next row this value is in unix time
#Buying
Close_buyadataAdjustor = 0.000000 #Adjusts to adata latest data adjust in positive range
Low_buyadataAdjustor = 0.000000 #Adjusts to adata latest data adjust in positive range
High_buyadataAdjustor = 0.000000 #Adjusts to adata latest data adjust in positive range
buyVolume = float(Constantai_params.get('buyVolume', '0.01')) #The volume of trades to buy
buyStoploss = float(Constantai_params.get('buyStoploss', '0.0001')) #Stop loss for buy action
buyTakeProfit = float(Constantai_params.get('buyTakeProfit', '0.0001')) #Take profit for buy action
buyMagic = int(Constantai_params.get('buyMagic', '123456')) # can identify
#Selling
Close_selladataAdjustor = -0.000000 #Adjusts to adata latest data adjust in negative range
Low_selladataAdjustor = -0.000000 #Adjusts to adata latest data adjust in negative range
High_selladataAdjustor = -0.000000 #Adjusts to adata latest data adjust in negative range
sellVolume = float(Constantai_params.get('sellVolume', '0.01')) #The volume of trades to sell
sellStoploss = float(Constantai_params.get('sellStoploss', '0.0001')) #Stop loss for sell action
sellTakeProfit = float(Constantai_params.get('sellTakeProfit', '0.0001')) #Take profit for sell action
sellMagic = int(Constantai_params.get('sellMagic', '123456')) # can identify


# Continue running the script
logging.basicConfig(filename='EURUSD/EURUSD.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
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
start_time = end_time - dt.timedelta(passedtimeConstantai)
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
        # Insert a new row with the data
        values = [timestamp, float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7])]
        cursor.execute("INSERT INTO EURUSDAdata (timestamp, [open], high, low, [close], tick_volume, spread, real_volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
        conn.commit()
        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.debug("Latest written Timeframe data")
        logging.debug(values)
        logging.debug("Rates Updated")

    # Check if this timestamp is the latest in the database
    cursor.execute("SELECT MAX(timestamp) FROM EURUSDAdata")
    latest_timestamp = cursor.fetchone()[0]
    if timestamp == latest_timestamp:
        # Update the row with the data
        values = [float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7]), timestamp]
        cursor.execute("UPDATE EURUSDAdata SET [open] = ?, high = ?, low = ?, [close] = ?, tick_volume = ?, spread = ?, real_volume = ? WHERE timestamp = ?", tuple(values))
        conn.commit()
        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.debug("Latest Timeframe data Updated")
        logging.debug(values)


conn.commit()
cursor.close()

# Execute the SQL update query to replace nulls with 0 in the specified column
cursor = conn.cursor()
#query = '''
#        UPDATE EURUSDAdata
#        SET [open] = ISNULL([open], 0),
#            high = ISNULL(high, 0),
#            low = ISNULL(low, 0),
#            [close] = ISNULL([close], 0),
#            pred_open = ISNULL(pred_open, 0),
#            pred_high = ISNULL(pred_high, 0),
#            pred_low = ISNULL(pred_low, 0),
#            pred_close = ISNULL(pred_close, 0)
#        '''
#cursor.execute(query)
#conn.commit()
    
# Load data from database
query = "SELECT TOP 2 timestamp, [open], high, low, [close], pred_open, pred_high, pred_low, pred_close FROM EURUSDAdata ORDER BY timestamp DESC"
cursor.execute(query)
rows = cursor.fetchall()
print("rows before")
print(rows)
rows = rows [:-1]
print("rows before")
print(rows)
rows = rows [::-1]
print("rows before")
print(rows)

# Skip any None or NULL values

for row in rows:
    skip_row = False
    for value in row:
        if value is None:
            skip_row = True
            break
    if skip_row:
        continue
        # Check for null or None values in rows 5 to 8
    if None in row[5:9]:
        continue
#for row in rows:
#    for i in range(len(row)):
#        if row[i] is None or row[i] == "NULL":
#           continue

    # Calculate the evaluation values

    
    if rows[0][1] is not None and rows[0][5] is not None:
        eval_open = rows[0][1] - rows[0][5]
    else:
        eval_open = 1.0

    if rows[0][2] is not None and rows[0][6] is not None:
        eval_high = rows[0][2] - rows[0][6]
    else:
        eval_high = 1.0
    
    if rows[0][3] is not None and rows[0][7] is not None:
        eval_low = rows[0][3] - rows[0][7]
    else:
        eval_low = 1.0
    
    if rows[0][4] is not None and rows[0][8] is not None:
        eval_close = rows[0][4] - rows[0][8]
    else:
        eval_close = 1.0
    print(eval_open)

    #if eval_open == 0:
    #    eval_open_reward = 1
    #eval_open_reward = -1
    #if eval_high == 0:
    #    eval_high_reward = 1
    #eval_high_reward = -1
    #if eval_low == 0:
    #    eval_low_reward = 1
    #eval_low_reward = -1
    #if eval_close == 0:
    #    eval_close_reward = 1
    #eval_close_reward = -1


    # Update the row with the evaluation values
    query = "UPDATE EURUSDAdata SET eval_open=?, eval_high=?, eval_low=?, eval_close=? WHERE timestamp=?"
    cursor.execute(query, (eval_open, eval_high, eval_low, eval_close, row[0]))
    conn.commit()

cursor.close()


#cursor = conn.cursor()
#for rate in rates:
#    timestamp = int(rate[0])

#    # Check if timestamp already exists in the database
#    cursor.execute("SELECT COUNT(*) FROM EURUSDAdata WHERE timestamp = ?", (timestamp,))
#    count = cursor.fetchone()[0]
#    if count == 0:
#         # Write the data to the database
#         values = [timestamp, float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7])]
#         cursor.execute("INSERT INTO EURUSDAdata (timestamp, [open], high, low, [close], tick_volume, spread, real_volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
#         #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#         logging.debug("Latest written Timeframe data")
#         logging.debug(values)
#         logging.debug("Rates Updated")

#cursor.commit()





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
to_date = dt.datetime.now() + dt.timedelta(hours=4)
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
profit_data = deals 
# create a dictionary to store profit data with timestamps as keys
profit_dict = {}
for profit_row in profit_data:
    profit_dict[profit_row[0]] = profit_row[2]

# insert deals data into the table
deals_data = []
for deal in deals:
    # execute a SELECT query to get the id from EURUSDAdata table
    cursor.execute("SELECT id FROM EURUSDAdata WHERE timestamp <= ? AND ? < timestamp + 60", (deal.time, deal.time))
    result = cursor.fetchone()
    if result is not None:
        # assign reward based on profit value
        if deal.profit < 0:
            reward = -1
        elif deal.profit == 0:
            reward = 0
        else:
            reward = 1
        # update the row with the new data
        cursor.execute("UPDATE EURUSDAdata SET ticket=?, [order]=?, time=?, type=?, entry=?, magic=?, position_id=?, reason=?, volume=?, price=?, commission=?, swap=?, profit=?, fee=?, symbol=?, comment=?, external_id=?, reward=? WHERE id=?", (deal.ticket, deal.order, deal.time, deal.type, deal.entry, deal.magic, deal.position_id, deal.reason, deal.volume, deal.price, deal.commission, deal.swap, deal.profit, deal.fee, deal.symbol, deal.comment, deal.external_id, reward, result[0]))
        # logging
        logging.debug("Deal data updated for time: " + str(deal.time))
    else: 
        # logging
        logging.debug("Complete WL update")
conn.commit()
conn.close()

# insert deals data into the table
#deals_data = []
#for deal in deals:
#    # execute a SELECT query to check if the time exists in the table
#    cursor.execute(f"SELECT time FROM EURUSDWLdata WHERE time = {deal.time}")
#    result = cursor.fetchone()
#    if result is None:
#        # assign reward based on profit value
#        if deal.profit < 0:
#            reward = -1
#        elif deal.profit == 0:
#            reward = 0
#        else:
#            reward = 1
#       # append data to the list
#        deal_data = [deal.ticket, deal.order, deal.time, deal.type, deal.entry, deal.magic, deal.position_id, deal.reason, deal.volume, deal.price, deal.commission, deal.swap, deal.profit, deal.fee, deal.symbol, deal.comment, deal.external_id, reward]
#        deals_data.append(deal_data)
#        # execute the INSERT statement with the tuple
#        cursor.execute("INSERT INTO EURUSDWLdata (ticket, [order], time, type, entry, magic, position_id, reason, volume, price, commission, swap, profit, fee, symbol, comment, external_id, reward) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(deal_data))
#        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#        logging.debug(deal_data)
#logging.debug("Deal data updated")
#conn.commit()
#conn.close()

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


# Execute the SQL update query to replace nulls with 0 in the specified column
cursor = conn.cursor()
query = 'UPDATE EURUSDAdata SET reward = 0 WHERE reward IS NULL'
cursor.execute(query)
conn.commit()

# Load data from database
query = f"SELECT TOP ({constantaiRowselector}) timestamp, [open], high, low, [close], eval_open, eval_high, eval_low, eval_close, reward FROM EURUSDAdata ORDER BY timestamp DESC"
cursor = conn.cursor()
cursor.execute(query)
data = cursor.fetchall()
for row in cursor:
    data.append(list(row))

# Close the database connection
cursor.close()
conn.close()

#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Rates data raw")
logging.debug(data)


# Convert data to numpy array and reverse the order of the rows
data = np.array(data[::-1])


#scaler = MinMaxScaler(feature_range=(0, 1))
#data = scaler.fit_transform(data)
X = data[:-1, 1:5]  # timestamp, [open], high, low, [close]
Y = np.roll(data[:-1, 1:5], -1, axis=0) # Shift the Y values by one time step to predict the next set of datapoints
reward = data[:, 9:10]
for i in range(len(reward)):
    for j in range(len(reward[i])):
        if reward[i][j] is None:
            reward[i][j] = -1

reward0 = data[:, 5:6]
for i in range(len(reward0)):
    for j in range(len(reward0[i])):
        if reward0[i][j] is None:
            reward0[i][j] = -1

reward1 = data[:, 6:7]
for i in range(len(reward1)):
    for j in range(len(reward1[i])):
        if reward1[i][j] is None:
            reward1[i][j] = -1

reward2 = data[:, 7:8]
for i in range(len(reward2)):
    for j in range(len(reward2[i])):
        if reward2[i][j] is None:
            reward2[i][j] = -1

reward3 = data[:, 8:9]
for i in range(len(reward3)):
    for j in range(len(reward3[i])):
        if reward3[i][j] is None:
            reward3[i][j] = -1

#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Input data <X>")
logging.debug(X)
logging.debug("Output data <Y>")
logging.debug(Y)
logging.debug("Profit data <profit>")
logging.debug(reward)

X = X.astype('float32')
Y = Y.astype('float32')
reward = reward.astype('float32')
reward0 = reward0.astype('float32')
reward1 = reward1.astype('float32')
reward2 = reward2.astype('float32')
reward3 = reward3.astype('float32')

#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Input data <X>.astype")
logging.debug(X)
logging.debug("Output data <Y>.astype")
logging.debug(Y)
logging.debug("Profit data <profit>.astype")
logging.debug(reward)


# Split the data into training and testing sets
split = int((constantaiTrainsplit) * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

R_train, R_test = reward[:split], reward[split:]
R_train = np.sign(R_train)  # Convert to -1, 0, or 1
R_train = (R_train + 1) % 2  # Convert to 0 or 1
R_test = np.sign(R_test)  # Convert to -1, 0, or 1
R_test = (R_test + 1) % 2  # Convert to 0 or 1

R0_train, R0_test = reward0[:split], reward0[split:]
R0_train = np.sign(R0_train)  # Convert to -1, 0, or 1
R0_train = (R0_train + 1) % 2  # Convert to 0 or 1
R0_test = np.sign(R0_test)  # Convert to -1, 0, or 1
R0_test = (R0_test + 1) % 2  # Convert to 0 or 1

R1_train, R1_test = reward1[:split], reward1[split:]
R1_train = np.sign(R1_train)  # Convert to -1, 0, or 1
R1_train = (R1_train + 1) % 2  # Convert to 0 or 1
R1_test = np.sign(R1_test)  # Convert to -1, 0, or 1
R1_test = (R1_test + 1) % 2  # Convert to 0 or 1

R2_train, R2_test = reward2[:split], reward2[split:]
R2_train = np.sign(R2_train)  # Convert to -1, 0, or 1
R2_train = (R2_train + 1) % 2  # Convert to 0 or 1
R2_test = np.sign(R2_test)  # Convert to -1, 0, or 1
R2_test = (R2_test + 1) % 2  # Convert to 0 or 1

R3_train, R3_test = reward3[:split], reward3[split:]
R3_train = np.sign(R3_train)  # Convert to -1, 0, or 1
R3_train = (R3_train + 1) % 2  # Convert to 0 or 1
R3_test = np.sign(R3_test)  # Convert to -1, 0, or 1
R3_test = (R3_test + 1) % 2  # Convert to 0 or 1

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
#model.fit(X_train, Y_train, epochs=constantaiEpochs, batch_size=constantaiBatchsize, sample_weight=R_train,
#          validation_data=(X_test, Y_test), validation_steps=len(X_test),
#          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)])
model.fit(X_train, Y_train, epochs=constantaiEpochs, batch_size=constantaiBatchsize,
          sample_weight=[R_train, R0_train, R1_train, R2_train, R3_train],
          validation_data=(X_test, Y_test),
          validation_steps=len(X_test),
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



#**********************************************************************************************************************************
#reads lastpred values and writes to db *******************************************************************************************
#**********************************************************************************************************************************



# Define the SQL statement to update the row with the predicted values
query_insert_row = "INSERT INTO EURUSDAdata (timestamp, pred_time, pred_open, pred_high, pred_low, pred_close) VALUES (?, ?, ?, ?, ?, ?)"
values = [int(next_timestamp), int(next_timestamp), float(Pred_Open), float(Pred_High), float(Pred_Low), float(Pred_Close)]
cursor.execute(query_insert_row, tuple(values))
#logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
logging.debug("Prediction values witten to EURUSDAdata")
logging.debug(values)

conn.commit()
conn.close()







# Do something with the predictions
Close_Decision_Adjustor_Buy = Close_buyadataAdjustor
Low_Decision_Adjustor_Buy = Low_buyadataAdjustor
High_Decision_Adjustor_Buy = High_buyadataAdjustor
Last_Close_Buy_Helper = Last_Close + Close_Decision_Adjustor_Buy
Last_Low_Buy_Helper = Last_Low + Low_Decision_Adjustor_Buy
Last_High_Buy_Helper = Last_High + High_Decision_Adjustor_Buy
if np.any(Pred_Close >= Last_Close_Buy_Helper) and (Pred_Low >= Last_Low_Buy_Helper) or (Pred_High >= Last_High_Buy_Helper) and (Pred_Close >= Last_Close_Buy_Helper):
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
    
    # set the maximum number of attempts to execute the order
    max_attempts = 3
    attempts = 0
    order_executed = False
    
    while not order_executed and attempts < max_attempts:
        
        # create a request for a new order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": type,
            "price": mt5.symbol_info_tick(symbol).ask,
            "sl": price - stop_loss,
            "tp": price + take_profit,
            "magic": magic_number,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # send the order request
        result = mt5.order_send(request)

        # check if the order was executed successfully
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            logging.debug("order executed with order_id={}".format(result.order))
            logging.debug("BUY")
            print("BUYING")
            order_executed = True
        else:
            #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            logging.error("order failed with retcode={}".format(result.retcode))
            logging.error("message={}".format(result.comment))
            print("BUY ORDER FAILED")
            print("order failed with retcode={}".format(result.retcode))
            print("message={}".format(result.comment))
            attempts += 1

    # check if the order was not executed after the maximum number of attempts
    if not order_executed:
        logging.error("maximum number of attempts to execute the order reached")
        print("maximum number of attempts to execute the order reached")

else:
    Close_Decision_Adjustor_Sell = Close_selladataAdjustor
    Low_Decision_Adjustor_Sell = Low_selladataAdjustor
    High_Decision_Adjustor_Sell = High_selladataAdjustor
    Last_Close_Sell_Helper = Last_Close + Close_Decision_Adjustor_Sell
    Last_Low_Sell_Helper = Last_Low + Low_Decision_Adjustor_Sell
    Last_High_Sell_Helper = Last_High + High_Decision_Adjustor_Sell
    if np.any(Pred_Close <= Last_Close_Sell_Helper) and (Pred_Low <= Last_Low_Sell_Helper) or (Pred_High <= Last_High_Sell_Helper) and (Pred_Close <= Last_Close_Sell_Helper):
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

        # set the maximum number of attempts to execute the order
        max_attempts = 3
        attempts = 0
        order_executed = False
    
        while not order_executed and attempts < max_attempts:
            # create a request for a new order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": type,
                "price": mt5.symbol_info_tick(symbol).bid,
                "sl": price + stop_loss,
                "tp": price - take_profit,
                "magic": magic_number,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # send the order request
            result = mt5.order_send(request)
            
            # check if the order was executed successfully
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                logging.debug("order executed with order_id={}".format(result.order))
                logging.debug("SELL")
                print("SELLING")
                order_executed = True    
            else:
                #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                logging.debug("order failed with retcode={}".format(result.retcode))
                logging.debug("message={}".format(result.comment))
                print("SELL ORDER FAILED")
                print("order failed with retcode={}".format(result.retcode))
                print("message={}".format(result.comment))
                attempts += 1

        # check if the order was not executed after the maximum number of attempts
        if not order_executed:
            logging.error("maximum number of attempts to execute the order reached")
            print("maximum number of attempts to execute the order reached")        

    else:
        #logging    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.debug("Do nothing")
        print("DO NOTHING!!!")
        # do nothing code here
        mt5.shutdown
time.sleep(10)
