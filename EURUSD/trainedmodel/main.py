import time
import subprocess

# Run the script
#script_path = "EURUSD/Trainmodel/EURUSD_tdata.py"
#subprocess.call(['python', script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)

# Set the total runtime for the program
total_runtime = 7200  # seconds

# Set the interval between script runs
script_interval = 60  # seconds

# Get the start time for the program
start_time = time.time()

# Start the loop to run the script
while True:
    # Check if the total runtime has elapsed
    if time.time() - start_time >= total_runtime:
        print("Program has ended.")
        break

    # Run the script
    script_path = "EURUSD/trainedmodel/EURUSD_adata.py"
    subprocess.Popen(['python', script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)

    # Wait for the interval
    time.sleep(script_interval)