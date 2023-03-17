from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sklearn
import MetaTrader5 as mt5
import pyodbc
import alpaca_trade_api as tradeapi
import datetime as dt
import decimal
import time
import os

# Connect to MT5
print("Establising Connection to MT5, Please wait")
mt5.initialize()
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1
print("Connection Successful")
print(symbol,"timeframe = " + str(timeframe))
#bars = 1000

# Calculate start and end times
end_time = dt.datetime.now()
start_time = end_time - dt.timedelta(seconds=80)
print("Data Time Start = " + str(start_time))
print("Data Time End = " + str(end_time))


# Get historical data
print("Getting historical data")
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
rates = np.array(rates)

# Print data to console
#for rate in rates:
#    print(rate)

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

X_scaler=MinMaxScaler()
Y_scaler=MinMaxScaler()
# Make predictions on new data
X_new = data[:, 1:9]
# Scale the new data point
X_new_scaled = X_scaler.fit_transform(X_new)
# Make a prediction on the scaled new data point
Y_pred = model.predict(X_new_scaled)


# Extract the file name data of the model loaded and print on screen
model_name = latest_model_file.split(".")[0]
timestamp = model_name.split("_")[-1]
print(f"Loaded model from file: {latest_model_file}, created at {timestamp}")
print("X_new")
print(X_new)
print("Y_pred")
print("Predicted closing value:", Y_pred[0][0])

# Do something with the predictions
if np.any(Y_pred > 0.95):

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
    print("Buy")

elif np.any(Y_pred < -0.95):
    # sell code here
    print("Sell")
else:
    # do nothing code here
    print("Do nothing")

# Scale the new data point
Y_pred_scaled = Y_scaler.fit_transform(Y_pred)
# Inverse scale the predicted value
Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)


