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
from datetime import datetime, timezone
import logging

logging.basicConfig(filename='EURUSD.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.info('\n''\nUpdating Information')



# Connect to MT5
logging.debug("Establising Connection to MT5, Please wait")
mt5.initialize()
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1
logging.debug("Connection Successful")
logging.debug("%s timeframe = %d", symbol, timeframe)



# Calculate start and end times
#end_time = dt.datetime(2023, 3, 1, 23, 59, 59)  # Set date
end_time = dt.datetime.now()
start_time = end_time - dt.timedelta(days=1)
logging.debug("Data Time Start = " + str(start_time))
logging.debug("Data Time End = " + str(end_time))



# Get historical data
logging.debug("Getting historical data")
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
rates = np.array(rates)




# Convert start_time and end_time to Unix timestamps
#start_time = int(start_time.replace(tzinfo=timezone.utc).timestamp())
#end_time = int(end_time.replace(tzinfo=timezone.utc).timestamp())


# Update start and end times
#start_time = end_time
#end_time = dt.datetime.now()

# Establish a connection to the SQL Express database
logging.debug("Establishing a connection to the SQL Express database")
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')
logging.debug("Connection established")

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
         logging.debug("Latest Timeframe data")
         logging.debug(values)

cursor.commit()

logging.debug("SQL complete MT data is up to date")

# call the other script
subprocess.run(['python', 'EURUSD/trainedmodel/EURUSD_wldata.py'])
