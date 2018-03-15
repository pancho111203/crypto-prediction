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
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Simulator crypto prices')
parser.add_argument('-r', '--realtime', action='store_const',
                   const=True, default=False,
                   help='Only visualize results of previously saved model')

args = parser.parse_args()

modelName = "model2.256lstmx2.stateful"

public_client = gdax.PublicClient()

class Simulator(object):
    def __init__(self, cryptoCoin='ETH', buyPercentage=0.5, initialUsd = 1000.0, initialCrypto = 1.0, isRealTime=True):
        self.initialCrypto = initialCrypto
        self.initialUsd = initialUsd
        self.isRealTime = isRealTime
        
        self.startTime = datetime.datetime.now().isoformat()
        self.usd = initialUsd
        self.crypto = initialCrypto
        self.cryptoCoin = cryptoCoin
        self.coinPair = '{}-USD'.format(self.cryptoCoin)
        self.buyPercentage = buyPercentage

        self.predictor = Predict_model(modelName)
        self.pastCurrentPrice = None
        self.pastPastCurrentPrice = None
        self.predictedDelta = None
        self.predictPrice = None

        self.valueHistory = []

        startingData = data_loader.getCandles('ETH-USD', 60, save=False)[['open']]

        for currentPrice in startingData.values:
            if self.pastCurrentPrice:
                self.predictor.get_model(self.pastCurrentPrice, currentPrice)

            self.pastPastCurrentPrice = self.pastCurrentPrice
            self.pastCurrentPrice = currentPrice

    def process(self, data):
        currPrice = data['price']
        time = data['time']
        if self.pastCurrentPrice:
            if self.predictPrice:
                err = currPrice - self.predictPrice
                logger.info('Previous prediction error: {}'.format(err))

# #            if self.pastPastCurrentPrice and abs(err)>0.09:
#             if self.pastPastCurrentPrice:
#                 self.predictor.training(self.pastPastCurrentPrice, self.pastCurrentPrice, currPrice)

            self.predictedDelta = self.predictor.get_model(self.pastCurrentPrice, currPrice)
            
            logger.info('Current Price: {}'.format(currPrice))
            self.predictPrice = self.predictedDelta*currPrice
            logger.info('Predicted Price: {}'.format(self.predictPrice))

            decision = self.predictor.buyer(self.predictedDelta)
            
            # TODO instead of taking price, take best bid for buys and best ask for sells
            if self.isRealTime:
                price = float(public_client.get_product_ticker(product_id=self.coinPair)['price'])
            else:
                price = currPrice
                
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

    if args.realtime:
        logging.basicConfig(level=logging.INFO)
        sim = Simulator()
        tickerFeed = TickerFeed(sim.coinPair, 60)
        tickerFeed.onTickerReceived(sim.process)
        tickerFeed.run()
    else:

        # sim = Simulator(isRealTime=False)
        # simTestData = data_loader.getCandles(sim.coinPair, 60, start=(datetime.datetime.now() - datetime.timedelta(days=3)).isoformat(), save=True)[['open']]
        # for (time, price) in simTestData.iterrows():
        #     price = price.item()
        #     time = time.isoformat()
        #     sim.process({
        #             'price': price,
        #             'time': time
        #     })
        logging.basicConfig(level=logging.WARN)
        sim = Simulator(isRealTime=False)
        with open(os.path.join(os.path.dirname(__file__), '../data/candles/ETH-USD_60_2018-02-14T00:00:25+01:00_2018-03-14T00:00:25+01:00.json'), 'r') as f:
            simTestData = pd.DataFrame(json.load(f))

        for (key, row) in simTestData.iterrows():
            time = pd.to_datetime(row[0], unit='s').isoformat()
            price = row[1]
            sim.process({
                    'price': price,
                    'time': time
            })