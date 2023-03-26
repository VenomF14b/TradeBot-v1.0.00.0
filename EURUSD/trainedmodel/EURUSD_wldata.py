import pyodbc
import MetaTrader5 as mt5
import datetime as dt
#from datetime import datetime
#from datetime import dt
import pandas as pd
import subprocess

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

 
# get the number of deals in history
# Calculate start and end times
#end_time = dt.datetime(2023, 3, 1, 23, 59, 59)  # Set date
to_date = dt.datetime.now() + dt.timedelta(hours=2)
from_date = to_date - dt.timedelta(days=14)

print(to_date)
print(from_date)
#from_date=datetime(2023,1,1)
#to_date=datetime.now()
# get deals for symbols whose names contain "EURUSD" within a specified interval
deals=mt5.history_deals_get(from_date, to_date, group="*EURUSD*")
# filter deals with zero profit
deals = [deal for deal in deals if deal.profit != 0]
# sort the dataframe by ticket in ascending order
#deals = sorted(deals, key=lambda deal: deal.ticket)

if deals==None:
    print("No deals with group=\"*EURUSD*\", error code={}".format(mt5.last_error()))
elif len(deals)> 0:
    print("history_deals_get({}, {}, group=\"*EURUSD*\")={}".format(from_date,to_date,len(deals)))


    # display these deals as a table using pandas.DataFrame
    df=pd.DataFrame(list(deals),columns=deals[0]._asdict().keys())
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(df)
#print("")

cursor = conn.cursor()

# create table to store deals data
#cursor.execute("CREATE TABLE EURUSDWLdata (ticket INT, [order] INT, time VARCHAR(255), type INT, entry FLOAT, magic INT, position_ID INT, reason INT, volume FLOAT, price FLOAT, commission FLOAT, swap FLOAT, profit FLOAT, fee FLOAT, symbol VARCHAR(50), comment VARCHAR(255), external_ID VARCHAR(255))")

# insert deals data into the table
for deal in deals:
    # execute a SELECT query to check if the position ID exists in the table
    cursor.execute(f"SELECT time FROM EURUSDWLdata WHERE time = {deal.time}")
    result = cursor.fetchone()
    if result is None:
        cursor.execute(f"INSERT INTO EURUSDWLdata VALUES ({deal.ticket}, {deal.order}, '{deal.time}', {deal.type}, {deal.entry}, {deal.magic}, {deal.position_id}, {deal.reason}, {deal.volume}, {deal.price}, {deal.commission}, {deal.swap}, {deal.profit}, {deal.fee}, '{deal.symbol}', '{deal.comment}', '{deal.external_id}')")
        print("Deal data updated")

# commit changes and close connection
conn.commit()
conn.close()

# call the other script
subprocess.run(['python', 'EURUSD/trainedmodel/EURUSD_constantai.py'])

