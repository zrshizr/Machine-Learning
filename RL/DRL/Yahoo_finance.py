import numpy as np
from matplotlib import pyplot as plt
from yahoo_finance import Share
import tensorflow as tf


def get_prices(share_symbol,start_date,end_date,cache_filename='stock_prices.npy'):
    try:
        stock_prices = np.load(cache_filename)
    except IOError:
        share = Share(share_symbol)
        stock_hist = share.get_historical(start_date,end_date)
        stock_prices = [stock_price['Open'] for stock_price in stock_hist]
        np.save(cache_filename,stock_prices)
    
    return stock_prices.astype(float)

def plot_prices(prices):
    plt.title('Opening stock prices')
    plt.xlabel('day')
    plt.ylabel('price')
    plt.plot(prices)        
    plt.savefig('prices.png')
    plt.show

if __name__ == '__main__':
    # prices = get_prices('YHOO','1992-07-22','2016-07-22') 
    # prices = get_prices('MSFT',"1992-07-22","2016-07-22")
     yahoo = Share('MSFT')

    # plot_prices(prices)