import MetaTrader5 as mt5
import numpy as np
import tensorflow as tf
import pyodbc
import datetime as dt
import decimal
import subprocess
import time
import ctypes
import signal
import sys
import os
from tensorflow import keras
from tensorflow.keras import layers






# Connect to MT5
print("Establising Connection to MT5, Please wait")
mt5.initialize()
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1
print("Connection Successful")
print(symbol,"timeframe = " + str(timeframe))



# Calculate start and end times
#end_time = dt.datetime(2023, 3, 1, 23, 59, 59)  # Set date
end_time = dt.datetime.now()
start_time = end_time - dt.timedelta(days=1)
print("Data Time Start = " + str(start_time))
print("Data Time End = " + str(end_time))



# Get historical data
print("Getting historical data")
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
rates = np.array(rates)

# Update start and end times
#start_time = end_time
#end_time = dt.datetime.now()

# Establish a connection to the SQL Express database
print("Establishing a connection to the SQL Express database")
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')
print("Connection established")

# Write the data to the database
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
         
print("database updated with the following data")
print(values)
conn.commit()
print("SQL complete MT data is up to date")

# call the other script
subprocess.run(['python', 'EURUSD/trainedmodel/EURUSD_constantai.py'])
