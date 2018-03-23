# -*- coding: utf-8 -*-
import logging
import os
logger = logging.getLogger(__name__)

import datetime
from dateutil.parser import parse as dateparse
import gdax
import math
import json
import pandas as pd
import time

public_client = gdax.PublicClient()

#start	Start time in ISO 8601
#end	End time in ISO 8601
#granularity in seconds
#The granularity field must be one of the following values: {60, 300, 900, 3600, 21600, 86400}. 
#Otherwise, your request will be rejected. 
#These values correspond to timeslices representing one minute, five minutes, fifteen minutes, one hour, six hours, and one day, respectively.
def getCandles(coinPair, granularity, start=None, end=None, save=True):
    def requestCandles(startTime_, endTime_):
        logger.info('Getting data from {} to {}'.format(startTime_.isoformat(), endTime_.isoformat()))
        for attempt in range(0, 100):   
            if attempt > 0:
                logger.debug('Attempt {}'.format(attempt+1))
            try:
                candles = public_client.get_product_historic_rates(coinPair, granularity=granularity, start=startTime_.isoformat(), end=endTime_.isoformat())
            except:
                candles = None
            if isinstance(candles, list):
                candles.reverse()
                return candles
            else:
                logger.error('ERROR requesting candles: {}'.format(candles))
                logger.error('Trying again')
                time.sleep(5)
                
        return []

    maxResultsPerCall = 300 # described in https://docs.gdax.com/#get-historic-rates
    secondsCoveredPerCall = maxResultsPerCall * granularity
    
    if granularity != 60 and granularity != 300 and granularity != 900 and granularity != 3600 and granularity != 21600 and granularity != 86400:
        logger.error('Invalid granularity({}), must be one of the following values: {60, 300, 900, 3600, 21600, 86400}'.format(granularity))
        return []
    
    if not end:
        endTime = datetime.datetime.now()
        end = endTime.isoformat()
    else:
        endTime = dateparse(end)

    if not start:
        startTime = endTime - datetime.timedelta(seconds=secondsCoveredPerCall-1)
    else:
        startTime = dateparse(start)
        start = startTime.isoformat()
    
    if endTime < startTime:
        logger.error('Starting date ({}) is after ending date ({})'.format(start, end))
        return []
    
    filename = os.path.join(os.path.dirname(__file__), 'data/candles/{}_{}_{}_{}.json'.format(coinPair, granularity, start, end))
    if os.path.isfile(filename):
        logger.info('Loading data from file: {}'.format(filename))
        with open(filename, 'r') as f:
            allCandles = json.load(f)
        
    else:            
        expectedResultsLength = int((endTime - startTime).total_seconds() / granularity) + 1
        numberOfRequests = math.ceil(expectedResultsLength / maxResultsPerCall)
        logger.debug('expectedResultsLength: {}'.format(expectedResultsLength))
        logger.debug('numberOfRequests: {}'.format(numberOfRequests))
        
        allCandles = []
        currentStartTime = startTime
        for i in range(0, numberOfRequests - 1):
            currentEndTime = currentStartTime + datetime.timedelta(seconds=secondsCoveredPerCall)
    
            allCandles += requestCandles(currentStartTime, currentEndTime)
                
            currentStartTime = currentEndTime + datetime.timedelta(seconds=granularity)
            
        # last request only takes until endTime
        allCandles += requestCandles(currentStartTime, endTime)
        
        if save is True:
            if not(os.path.exists(os.path.join(os.path.dirname(__file__), 'data/candles'))):
                os.makedirs(os.path.join(os.path.dirname(__file__), 'data/candles'))
    
            logger.info('Saving data on file: {}'.format(filename))
            with open(filename, 'a+') as f:
                json.dump(allCandles, f)

    candlesFrame = pd.DataFrame(allCandles)
    candlesFrame.index = pd.to_datetime(candlesFrame[0], unit='s')
    del candlesFrame[0]
    candlesFrame.columns = ['low', 'high', 'open', 'close', 'volume' ]

    return candlesFrame


if __name__ == '__main__':  
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler())
    candles = getCandles('ETH-USD', granularity=60, start='2018-02-27T12:50:25+01:00', end='2018-02-28T12:50:25+01:00', save=True)
