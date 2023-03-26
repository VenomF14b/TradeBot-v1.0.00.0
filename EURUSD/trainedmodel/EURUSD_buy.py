import time
import MetaTrader5 as mt5
import os
import sys
import logging

logging.basicConfig(filename='EURUSD.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.info("Trade information (BUY)")

# connect to MetaTrader 5
if not mt5.initialize():
    logging.error("initialize() failed")
    mt5.shutdown()

# define the symbol and order type
symbol = "EURUSD"
lot_size = 1.0
stop_loss = 0.0001
take_profit = 0.0001
magic_number = 123456
price = mt5.symbol_info_tick(symbol).ask
type = mt5.ORDER_TYPE_BUY


# Do something with the price
logging.debug(f"The latest price for {symbol} is {price}.")
    # create a request for a new order
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot_size,
    "type": type,
    "price": price,
    "sl": price - stop_loss,
    "tp": price + take_profit,
    "magic": magic_number,
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

# send the order request
#result = mt5.orders_send(request)
result = mt5.order_send(request)

# check if the order was executed successfully
if result.retcode != mt5.TRADE_RETCODE_DONE:
    logging.error("order failed with retcode={}".format(result.retcode))
    logging.error("message={}".format(result.comment))
    print("BUY ORDER FAILED")
    print("order failed with retcode={}".format(result.retcode))
    print("message={}".format(result.comment))
else:
    logging.debug("order executed with order_id={}".format(result.order))



# disconnect from MetaTrader 5
mt5.shutdown()
logging.debug("BUY")
print("BUY")

time.sleep(10)