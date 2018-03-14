# -*- coding: utf-8 -*-
import logging
import os
import sys
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
___file___ = __file__
import json
from TickerFeed import TickerFeed
import data_loader
from model1 import Predict_model
import gdax
import datetime
modelName = "model2.256lstmx2.stateful"

class Simulator(object):
    def __init__(self, cryptoCoin='ETH', buyPercentage=0.5, initialUsd = 1000.0, initialCrypto = 1.0):
        self.initialCrypto = initialCrypto
        self.initialUsd = initialUsd

        self.startTime = datetime.datetime.now().isoformat()
        self.usd = initialUsd
        self.crypto = initialCrypto
        self.public_client = gdax.PublicClient()
        self.cryptoCoin = cryptoCoin
        self.coinPair = '{}-USD'.format(self.cryptoCoin)
        self.buyPercentage = buyPercentage
        self.tickerFeed = TickerFeed(self.coinPair, 60)

        self.tickerFeed.onTickerReceived(self.process)
        self.predictor = Predict_model(modelName)
        self.pastCurrentPrice = None
        self.pastPastCurrentPrice = None
        self.predictedDelta = None
        self.predictPrice = None

        self.valueHistory = []

        startingData = data_loader.getCandles('ETH-USD', 60)[['open']]

        for currentPrice in startingData.values:
            if self.pastCurrentPrice:
                self.predictor.get_model(self.pastCurrentPrice, currentPrice)

            self.pastPastCurrentPrice = self.pastCurrentPrice
            self.pastCurrentPrice = currentPrice

    def run(self):
        self.tickerFeed.run()
        
    def process(self, data):
        currPrice = data['price']
        time = data['time']
        if self.pastCurrentPrice:
            if self.predictPrice:
                err = currPrice - self.predictPrice
                logger.info('Previous prediction error: {}'.format(err))

#            if self.pastPastCurrentPrice and abs(err)>0.09:
            if self.pastPastCurrentPrice:
                self.predictor.training(self.pastPastCurrentPrice, self.pastCurrentPrice, currPrice)

            self.predictedDelta = self.predictor.get_model(self.pastCurrentPrice, currPrice)
            
            logger.info('Current Price: {}'.format(currPrice))
            self.predictPrice = self.predictedDelta*currPrice
            logger.info('Predicted Price: {}'.format(self.predictPrice))

            decision = self.predictor.buyer(self.predictedDelta)
            
            # TODO instead of taking price, take best bid for buys and best ask for sells
            price = float(self.public_client.get_product_ticker(product_id=self.coinPair)['price'])
            if decision == 'buy':
                self.buy(self.usd * self.buyPercentage, price)
                
            elif decision == 'sell':
                self.sell(self.crypto * self.buyPercentage, price)
                
            logger.info('Current Holdings:\nUSD: {}\n{}: {}'.format(self.usd, self.cryptoCoin, self.crypto))

            currentValue = (price * self.crypto) + self.usd
            initialValue = (price * self.initialCrypto) + self.initialUsd

            logger.info('Total USD Value: {}'.format((price * self.crypto) + self.usd))

            logger.info('Change: {}usd, {}%'.format(currentValue - initialValue, (currentValue / initialValue) * 100))
            self.valueHistory.append([time, currentValue])
            if len(self.valueHistory) > 0 and len(self.valueHistory) % 30 == 0:
                with open(os.path.join(os.path.dirname(___file___), "../prediction/checkpoint/{}/stats_{}.json".format(modelName, self.startTime)), 'w') as f:
                    json.dump(self.valueHistory, f)

        self.pastPastCurrentPrice = self.pastCurrentPrice
        self.pastCurrentPrice =  currPrice
        logger.info("")
        
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