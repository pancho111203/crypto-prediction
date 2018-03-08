# -*- coding: utf-8 -*-
import logging
import os
import sys
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from TickerFeed import TickerFeed
from model import Predict_model
import gdax

class Simulator(object):
    def __init__(self, cryptoCoin='ETH', buyPercentage=0.3):
        self.usd = 1000.0
        self.crypto = 1.0
        self.public_client = gdax.PublicClient()
        self.cryptoCoin = cryptoCoin
        self.coinPair = '{}-USD'.format(self.cryptoCoin)
        self.buyPercentage = buyPercentage
        self.tickerFeed = TickerFeed(self.coinPair, 60)
        self.tickerFeed.onTickerReceived(self.process)
        self.predictor = Predict_model("model1")
        
    def run(self):
        self.tickerFeed.run()
        
    def process(self, data):
        currPrice = data['price']
        print("Current Price", currPrice)
        predictedPrice = self.predictor.get_model(currPrice)
        print("Predict Price", predictedPrice)
        decision = self.predictor.buyer(currPrice, predictedPrice)
        
        # TODO instead of taking price, take best bid for buys and best ask for sells
        price = float(self.public_client.get_product_ticker(product_id=self.coinPair)['price'])
        if decision == 'buy':
            self.buy(self.usd * self.buyPercentage, price)
            
        elif decision == 'sell':
            self.sell(self.crypto * self.sellPercentage, price)
            
        logger.info('Current Holdings:\nUSD: {}\n{}: {}'.format(self.usd, self.cryptoCoin, self.crypto))
        
    def buy(self, amountInUsd, price):
        # TODO take into account exchange fees
        amountInCrypto = amountInUsd / price
        logger.info('Buying {} {} for {} USD ({})'.format(amountInCrypto, self.cryptoCoin, amountInUsd, price))

        self.usd -= amountInUsd
        self.crypto += amountInCrypto
        
    def sell(self, amountInCrypto, price):
        # TODO take into account exchange fees
        amountInUsd = amountInCrypto * price
        logger.info('Selling {} {} for {} USD ({})'.format(amountInCrypto, self.cryptoCoin, amountInUsd, price))

        self.usd += amountInUsd
        self.crypto -= cryptoAmount
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test = Simulator()
    test.run()