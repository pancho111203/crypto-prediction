# -*- coding: utf-8 -*-
import logging
import os
import sys
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import gdax
import threading


class TickerFeed(object):
    def __init__(self, coinPair, tickTimeInS=60):
        self.coinPair = coinPair
        self.tickTimeInS = tickTimeInS
        
        self.public_client = gdax.PublicClient()
        self.onTickerCallbacks = []
        
        self.timer = None
        
    def run(self):
        self.tick()
        self.timer = self._setInterval(self.tick, self.tickTimeInS)
        
    def onTickerReceived(self, cb):
        self.onTickerCallbacks.append(cb)
        
    def tick(self):
        ticker = self.public_client.get_product_ticker(product_id='ETH-USD')
        out = {
            'price': float(ticker['price']),
            'time': ticker['time']
        }
        
        for cb in self.onTickerCallbacks:
            cb(out)
        
    def _setInterval(self, func, sec):
        def func_wrapper():
            self._setInterval(func, sec)
            func()
        t = threading.Timer(sec, func_wrapper)
        t.start()
        return t


            
if __name__ == '__main__':
    test = TickerFeed('ETH-USD', 1)
    test.onTickerReceived(lambda x: print(x))
    test.run()