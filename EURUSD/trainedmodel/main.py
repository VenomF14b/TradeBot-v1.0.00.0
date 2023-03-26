import time
import subprocess
import MetaTrader5 as mt5

# initialize the MT5 connection
if not mt5.initialize():
    print("Failed to initialize MT5 connection!")
    exit()

# Set the total runtime for the program
total_runtime = 21600  # seconds

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
    script_path = "EURUSD/trainedmodel/EURUSD_adata.py"
    subprocess.Popen(['python', script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)

    # Wait for the interval
    time.sleep(script_interval)

# shutdown MT5 connection
mt5.shutdown()