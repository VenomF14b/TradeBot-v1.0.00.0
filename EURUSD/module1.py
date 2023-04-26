import tkinter as tk
import configparser
import logging

# Create logger for file1
EURUSDmodellog = logging.getLogger('EURUSDmodel')
file_handler1 = logging.FileHandler('EURUSD\EURUSDmodel.log')
EURUSDmodellog.addHandler(file_handler1)
EURUSDmodellog.setLevel(logging.DEBUG)

# Create logger for file2
EURUSDtdatalog = logging.getLogger('EURUSDtdata')
file_handler2 = logging.FileHandler('EURUSD\EURUSDtdata.log')
EURUSDtdatalog.addHandler(file_handler2)
EURUSDtdatalog.setLevel(logging.DEBUG)

# Log messages to the appropriate loggers
EURUSDmodellog.debug("This message is logged to file1.log")
EURUSDtdatalog.debug("This message is logged to file2.log")

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        self.create_widgets1()
        self.create_widgets2()


        # Set window size to match screen resolution
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_width}x{screen_height}")

    def create_widgets1(self):
        # Heading
        self.heading_label0 = tk.Label(self, text="Training Parameters:")
        self.heading_label0.grid(row=0, column=0, sticky='w')

        config0 = configparser.ConfigParser()
        config0.read('EURUSD/configEURUSD.ini')
        Conf_passedtime = config0.get('Training Parameters','passedtime')
        Conf_trainerairowselector = config0.get('Training Parameters','trainerairowselector')
        Conf_traineraiepochs = config0.get('Training Parameters','traineraiepochs')
        Conf_traineraibatchsize = config0.get('Training Parameters','traineraibatchsize')


        # userIn_passedtime
        self.userIn_passedTime_label = tk.Label(self, text="Historical data request range in days:")
        self.userIn_passedTime_label.grid(row=1, column=0, sticky='w')
        self.userIn_passedTime_entry = tk.Entry(self) 
        self.userIn_passedTime_entry.grid(row=2, column=0, sticky='w')
        self.userIn_passedTime_entry.insert(0, Conf_passedtime)

        # userIn_trainerairowselector
        self.userIn_traineraiRowselector_label = tk.Label(self, text="Historical data size number of rows @ 60 second intervals: ")
        self.userIn_traineraiRowselector_label.grid(row=3, column=0, sticky='w')
        self.userIn_traineraiRowselector_entry = tk.Entry(self)
        self.userIn_traineraiRowselector_entry.grid(row=4, column=0, sticky='w')
        self.userIn_traineraiRowselector_entry.insert(0, Conf_trainerairowselector)
        
        # userIn_traineraiepochs
        self.userIn_traineraiEpochs_label = tk.Label(self, text="Epochs: ")
        self.userIn_traineraiEpochs_label.grid(row=5, column=0, sticky='w')
        self.userIn_traineraiEpochs_entry = tk.Entry(self)
        self.userIn_traineraiEpochs_entry.grid(row=6, column=0, sticky='w')
        self.userIn_traineraiEpochs_entry.insert(0, Conf_traineraiepochs)
        
        # userIn_traineraibatchsize
        self.userIn_traineraiBatchsize_label = tk.Label(self, text="Batch size: ")
        self.userIn_traineraiBatchsize_label.grid(row=7, column=0, sticky='w')
        self.userIn_traineraiBatchsize_entry = tk.Entry(self)
        self.userIn_traineraiBatchsize_entry.grid(row=8, column=0, sticky='w')
        self.userIn_traineraiBatchsize_entry.insert(0, Conf_traineraibatchsize)


        # create submit button
        self.submit_button0 = tk.Button(self, text="Train new model", command=self.submit)
        self.submit_button0.grid(row=31, column=0, sticky='w')

    def submit(self):
        # get the user inputs from the entries
        userIn_passedTime = self.userIn_passedTime_entry.get()
        userIn_traineraiRowselector = self.userIn_traineraiRowselector_entry.get()
        userIn_traineraiEpochs = self.userIn_traineraiEpochs_entry.get()
        userIn_traineraiBatchsize = self.userIn_traineraiBatchsize_entry.get()


        try:
            userIn_passedTime = int(userIn_passedTime)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_traineraiRowselector = int(userIn_traineraiRowselector)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_traineraiEpochs = int(userIn_traineraiEpochs)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_traineraiBatchsize = int(userIn_traineraiBatchsize)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        
        # do something with the user inputs
                # Create a ConfigParser object, Set the values for the configuration file, Write the configuration to a file **********************
        #**********************************************************************************************************************************
        config0 = configparser.ConfigParser()
        config0['Training Parameters'] = {
            'passedtime': int(userIn_passedTime),
            'trainerairowselector': int(userIn_traineraiRowselector),
            'trainerairpochs': int(userIn_traineraiEpochs),
            'traineraibatchsize': int(userIn_traineraiBatchsize)
            }
        with open('EURUSD/EURUSD.ini', 'w') as configfile:
            config.write(configfile)
        #**********************************************************************************************************************************



    def create_widgets2(self):
        # Heading
        self.heading_label = tk.Label(self, text="Running Parameters:")
        self.heading_label.grid(row=0, column=1, sticky='w')

        config = configparser.ConfigParser()
        config.read('EURUSD/configEURUSDa.ini')
        Conf_passedtimeconstantai = config.get('Constantai Parameters','passedtimeconstantai')
        Conf_passedtimewldataconstantai = config.get('Constantai Parameters','passedtimewldataconstantai')
        Conf_constantairowselector = config.get('Constantai Parameters','constantairowselector')
        Conf_wldatarowselector = config.get('Constantai Parameters','wldatarowselector')
        Conf_constantaitrainsplit = config.get('Constantai Parameters','constantaitrainsplit')
        Conf_constantaiepochs = config.get('Constantai Parameters','constantaiepochs')
        Conf_constantaibatchsize = config.get('Constantai Parameters','constantaibatchsize')
        Conf_buyvolume = config.get('Constantai Parameters','buyvolume')
        Conf_buystoploss = config.get('Constantai Parameters','buystoploss')
        Conf_buytakeprofit = config.get('Constantai Parameters','buytakeprofit')
        Conf_buymagic = config.get('Constantai Parameters','buymagic')
        Conf_sellvolume = config.get('Constantai Parameters','sellvolume')
        Conf_sellstoploss = config.get('Constantai Parameters','sellstoploss')
        Conf_selltakeprofit = config.get('Constantai Parameters','selltakeprofit')
        Conf_sellmagic = config.get('Constantai Parameters','sellmagic')

        # create label and entry for userIn_passedtimeConstantai
        self.userIn_passedtimeConstantai_label = tk.Label(self, text="Historical data request range in days:")
        self.userIn_passedtimeConstantai_label.grid(row=1, column=1, sticky='w')
        self.userIn_passedtimeConstantai_entry = tk.Entry(self)
        self.userIn_passedtimeConstantai_entry.grid(row=2, column=1, sticky='w')
        self.userIn_passedtimeConstantai_entry.insert(0, Conf_passedtimeconstantai)

        # create label and entry for userIn_passedtimeWLdataConstantai
        self.userIn_passedtimeWLdataConstantai_label = tk.Label(self, text="win loss data request range in days:")
        self.userIn_passedtimeWLdataConstantai_label.grid(row=3, column=1, sticky='w')
        self.userIn_passedtimeWLdataConstantai_entry = tk.Entry(self)
        self.userIn_passedtimeWLdataConstantai_entry.grid(row=4, column=1, sticky='w')
        self.userIn_passedtimeWLdataConstantai_entry.insert(0, Conf_passedtimewldataconstantai)

        # create label and entry for userIn_constantaiRowselector
        self.userIn_constantaiRowselector_label = tk.Label(self, text="Historical data size number of rows @ 60 second intervals: ")
        self.userIn_constantaiRowselector_label.grid(row=5, column=1, sticky='w')
        self.userIn_constantaiRowselector_entry = tk.Entry(self)
        self.userIn_constantaiRowselector_entry.grid(row=6, column=1, sticky='w')
        self.userIn_constantaiRowselector_entry.insert(0, Conf_constantairowselector)

        # create label and entry for userIn_wldataRowselector
        self.userIn_wldataRowselector_label = tk.Label(self, text="Win loss size number of rows:")
        self.userIn_wldataRowselector_label.grid(row=7, column=1, sticky='w')
        self.userIn_wldataRowselector_entry = tk.Entry(self)
        self.userIn_wldataRowselector_entry.grid(row=8, column=1, sticky='w')
        self.userIn_wldataRowselector_entry.insert(0, Conf_wldatarowselector)

        # create label and entry for userIn_constantaiTrainsplit
        self.userIn_constantaiTrainsplit_label = tk.Label(self, text="data split example '0.70' 70/30 training/testing:")
        self.userIn_constantaiTrainsplit_label.grid(row=9, column=1, sticky='w')
        self.userIn_constantaiTrainsplit_entry = tk.Entry(self)
        self.userIn_constantaiTrainsplit_entry.grid(row=10, column=1, sticky='w')
        self.userIn_constantaiTrainsplit_entry.insert(0, Conf_constantaitrainsplit)

        # create label and entry for userIn_constantaiEpochs
        self.userIn_constantaiEpochs_label = tk.Label(self, text="Epochs: ")
        self.userIn_constantaiEpochs_label.grid(row=11, column=1, sticky='w')
        self.userIn_constantaiEpochs_entry = tk.Entry(self)
        self.userIn_constantaiEpochs_entry.grid(row=12, column=1, sticky='w')
        self.userIn_constantaiEpochs_entry.insert(0, Conf_constantaiepochs)

        # userIn_constantaiBatchsize
        self.userIn_constantaiBatchsize_label = tk.Label(self, text="Batch size:")
        self.userIn_constantaiBatchsize_label.grid(row=13, column=1, sticky='w')
        self.userIn_constantaiBatchsize_entry = tk.Entry(self)
        self.userIn_constantaiBatchsize_entry.grid(row=14, column=1, sticky='w')
        self.userIn_constantaiBatchsize_entry.insert(0, Conf_constantaibatchsize)

        # userIn_buyVolume
        self.userIn_buyVolume_label = tk.Label(self, text="Set buy volume example 0.01: ")
        self.userIn_buyVolume_label.grid(row=15, column=1, sticky='w')
        self.userIn_buyVolume_entry = tk.Entry(self)
        self.userIn_buyVolume_entry.grid(row=16, column=1, sticky='w')
        self.userIn_buyVolume_entry.insert(0, Conf_buyvolume)

        # userIn_buystoploss
        self.userIn_buyStoploss_label = tk.Label(self, text="Set buy StopLoss example 0.0001: ")
        self.userIn_buyStoploss_label.grid(row=17, column=1, sticky='w')
        self.userIn_buyStoploss_entry = tk.Entry(self)
        self.userIn_buyStoploss_entry.grid(row=18, column=1, sticky='w')
        self.userIn_buyStoploss_entry.insert(0, Conf_buystoploss)

        # userIn_buytakeprofit
        self.userIn_buyTakeprofit_label = tk.Label(self, text="Set buy TakeProfit example 0.0001: ")
        self.userIn_buyTakeprofit_label.grid(row=19, column=1, sticky='w')
        self.userIn_buyTakeprofit_entry = tk.Entry(self)
        self.userIn_buyTakeprofit_entry.grid(row=20, column=1, sticky='w')
        self.userIn_buyTakeprofit_entry.insert(0, Conf_buytakeprofit)

        # userIn_buymagic
        self.userIn_buyMagic_label = tk.Label(self, text="Set 6 digit buy identifier buy action: ")
        self.userIn_buyMagic_label.grid(row=21, column=1, sticky='w')
        self.userIn_buyMagic_entry = tk.Entry(self)
        self.userIn_buyMagic_entry.grid(row=22, column=1, sticky='w')
        self.userIn_buyMagic_entry.insert(0, Conf_buymagic)

        # userIn_sellvolume
        self.userIn_sellVolume_label = tk.Label(self, text="Set Sell volume example 0.01: ")
        self.userIn_sellVolume_label.grid(row=23, column=1, sticky='w')
        self.userIn_sellVolume_entry = tk.Entry(self)
        self.userIn_sellVolume_entry.grid(row=24, column=1, sticky='w')
        self.userIn_sellVolume_entry.insert(0, Conf_sellvolume)

        # userIn_sellstoploss
        self.userIn_sellStoploss_label = tk.Label(self, text="Set Sell StopLoss example 0.0001: ")
        self.userIn_sellStoploss_label.grid(row=25, column=1, sticky='w')
        self.userIn_sellStoploss_entry = tk.Entry(self)
        self.userIn_sellStoploss_entry.grid(row=26, column=1, sticky='w')
        self.userIn_sellStoploss_entry.insert(0, Conf_sellstoploss)

        # userIn_selltakeprofit
        self.userIn_sellTakeprofit_label = tk.Label(self, text="Set Sell TakeProfit example 0.0001: ")
        self.userIn_sellTakeprofit_label.grid(row=27, column=1, sticky='w')
        self.userIn_sellTakeprofit_entry = tk.Entry(self)
        self.userIn_sellTakeprofit_entry.grid(row=28, column=1, sticky='w')
        self.userIn_sellTakeprofit_entry.insert(0, Conf_selltakeprofit)

        # userIn_sellmagic
        self.userIn_sellMagic_label = tk.Label(self, text="Set 6 digit Sell identifier Sell action: ")
        self.userIn_sellMagic_label.grid(row=29, column=1, sticky='w')
        self.userIn_sellMagic_entry = tk.Entry(self)
        self.userIn_sellMagic_entry.grid(row=30, column=1, sticky='w')
        self.userIn_sellMagic_entry.insert(0, Conf_sellmagic)

        # create submit button
        self.submit_button = tk.Button(self, text="Start Trading", command=self.submit)
        self.submit_button.grid(row=31, column=1, sticky='w')

    def submit(self):
        # get the user inputs from the entries
        userIn_passedtimeConstantai = self.userIn_passedtimeConstantai_entry.get()
        userIn_passedtimeWLdataConstantai = self.userIn_passedtimeWLdataConstantai_entry.get()
        userIn_constantaiRowselector = self.userIn_constantaiRowselector_entry.get()
        userIn_wldataRowselector = self.userIn_wldataRowselector_entry.get()
        userIn_constantaiTrainsplit = self.userIn_constantaiTrainsplit_entry.get()
        userIn_constantaiEpochs = self.userIn_constantaiEpochs_entry.get()
        userIn_constantaiBatchsize = self.userIn_constantaiBatchsize_entry.get()
        userIn_buyVolume = self.userIn_buyVolume_entry.get()
        userIn_buyStoploss = self.userIn_buyStoploss_entry.get()
        userIn_buyTakeprofit = self.userIn_buyTakeprofit_entry.get()
        userIn_buyMagic = self.userIn_buyMagic_entry.get()
        userIn_sellVolume = self.userIn_sellVolume_entry.get()
        userIn_sellStoploss = self.userIn_sellStoploss_entry.get()
        userIn_sellTakeprofit = self.userIn_sellTakeprofit_entry.get()
        userIn_sellMagic = self.userIn_sellMagic_entry.get()



        # validate the user inputs
        try:
            userIn_passedtimeConstantai = int(userIn_passedtimeConstantai)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_passedtimeWLdataConstantai = int(userIn_passedtimeWLdataConstantai)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_constantaiRowselector = int(userIn_constantaiRowselector)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_wldataRowselector = int(userIn_wldataRowselector)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_constantaiTrainsplit = float(userIn_constantaiTrainsplit)
        except ValueError:
            print("Invalid input! Please enter a float value.")
            return
        try:
            userIn_constantaiEpochs = int(userIn_constantaiEpochs)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_constantaiBatchsize = int(userIn_constantaiBatchsize)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_buyVolume = float(userIn_buyVolume)
        except ValueError:
            print("Invalid input! Please enter a float value.")
            return
        try:
            userIn_buyStoploss = float(userIn_buyStoploss)
        except ValueError:
            print("Invalid input! Please enter a float value.")
            return
        try:
            userIn_buyTakeprofit = float(userIn_buyTakeprofit)
        except ValueError:
            print("Invalid input! Please enter a float value.")
            return
        try:
            userIn_buyMagic = int(userIn_buyMagic)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        try:
            userIn_sellVolume = float(userIn_sellVolume)
        except ValueError:
            print("Invalid input! Please enter a float value.")
            return
        try:
            userIn_sellStoploss = float(userIn_sellStoploss)
        except ValueError:
            print("Invalid input! Please enter a float value.")
            return
        try:
            userIn_sellTakeprofit = float(userIn_sellTakeprofit)
        except ValueError:
            print("Invalid input! Please enter a float value.")
            return
        try:
            userIn_sellMagic = int(userIn_sellMagic)
        except ValueError:
            print("Invalid input! Please enter an integer value.")
            return
        
        # do something with the user inputs
                # Create a ConfigParser object, Writes to the configuration file ******************************************************************
        #**********************************************************************************************************************************
        config = configparser.ConfigParser()
        config['Constantai Parameters'] = {
            'passedtimeConstantai': int(userIn_passedtimeConstantai),
            'passedtimewldataconstantai': int(userIn_passedtimeWLdataConstantai),
            'constantairowselector': int(userIn_constantaiRowselector),
            'wldatarowselector': int(userIn_wldataRowselector),
            'constantaitrainsplit': float(userIn_constantaiTrainsplit),
            'constantaiepochs': int(userIn_constantaiEpochs),
            'constantaibatchsize': int(userIn_constantaiBatchsize),
            'buyvolume': float(userIn_buyVolume),
            'buystoploss': float(userIn_buyStoploss),
            'buytakeProfit': float(userIn_buyTakeprofit),
            'buymagic': int(userIn_buyMagic),
            'sellvolume': float(userIn_sellVolume),
            'sellstoploss': float(userIn_sellStoploss),
            'selltakeprofit': float(userIn_sellTakeprofit),
            'sellmagic': int(userIn_sellMagic)
            }
        with open('EURUSD/EURUSDa.ini', 'w') as configfile:
            config.write(configfile)
    #**********************************************************************************************************************************

# create the application and run it
root = tk.Tk()
app = App(master=root)
app.mainloop()
