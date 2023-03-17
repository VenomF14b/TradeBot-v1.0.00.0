from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import numpy as np
import sklearn
import MetaTrader5 as mt5
import pyodbc
import alpaca_trade_api as tradeapi
import datetime as dt
import decimal
import time
import os

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
query = f"SELECT TOP 2 timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM Tdata00 ORDER BY timestamp DESC"
data = []
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
   data.append(row)
cursor.close()

# Convert data to numpy array and reverse the order of the rows
data = np.array(data[::-1])
X_new = data[:, 1:4]

# Make a prediction on the new data point
Y_pred = model.predict(X_new)

# Normalize the data usinf MinMax
#scaler = MinMaxScaler()
#X_new = scaler.fit_transform(X)
#Y_pred = scaler.fit_transform(Y)


# Normalize the data using z-score normalization
#X_new = zscore(X_new)
#Y_pred = zscore(Y_pred)

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


