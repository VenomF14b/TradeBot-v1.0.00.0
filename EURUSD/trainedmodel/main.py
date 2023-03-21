import time
import subprocess
import os
import signal

while True:
    
        script_path = "EURUSD/trainedmodel/EURUSD_adata.py"
        #process = os.system(f"start cmd /k python {script_path}")
        subprocess.Popen(['python', script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
        time.sleep(60)  # Wait for 60 seconds
