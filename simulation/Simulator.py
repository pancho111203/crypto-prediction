# -*- coding: utf-8 -*-
import logging
import os
import sys
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from TickerFeed import TickerFeed
from model1 import Predict_model
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
        self.predictor = Predict_model("model2")
        self.predictedPrice = None
        self.pastCurrentPrice = None
        
    def run(self):
        self.tickerFeed.run()
        
    def process(self, data):
        currPrice = data['price']

        if self.predictedPrice:
            logger.info('Previous prediction error: {}'.format(currPrice - self.predictedPrice))
            self.predictor.training(self.pastCurrentPrice, currPrice)

        self.predictedPrice = self.predictor.get_model(currPrice)
        self.pastCurrentPrice = currPrice
        
        logger.info('Current Price: {}'.format(currPrice))
        logger.info('Predicted Price: {}'.format(self.predictedPrice))

        decision = self.predictor.buyer(currPrice, self.predictedPrice)
        
        # TODO instead of taking price, take best bid for buys and best ask for sells
        price = float(self.public_client.get_product_ticker(product_id=self.coinPair)['price'])
        if decision == 'buy':
            self.buy(self.usd * self.buyPercentage, price)
            
        elif decision == 'sell':
            self.sell(self.crypto * self.buyPercentage, price)
            
        logger.info('Current Holdings:\nUSD: {}\n{}: {}'.format(self.usd, self.cryptoCoin, self.crypto))
        logger.info('Total USD Value: {}'.format((price * self.crypto) + self.usd))
        
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
        self.crypto -= amountInCrypto
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test = Simulator()
    test.run()