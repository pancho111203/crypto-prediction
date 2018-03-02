# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import json
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import data_loader

candles = data_loader.getCandles('ETH-USD', 60, '2018-02-25T00:00:25+01:00', '2018-02-25T23:58:25+01:00')
#1 hr moving average
movingAverage1hr = candles.close.rolling(60, min_periods=60).mean()

candlesChart = go.Candlestick(x=candles.index,
                       open=candles.open,
                       high=candles.high,
                       low=candles.low,
                       close=candles.close)

opens = go.Scatter(
    x=candles.index,
    y=candles.open,
    name='open'
)

movingAverage1hrPlot = go.Scatter(
    x=movingAverage1hr.index,
    y=movingAverage1hr.values,
    name='movingAverage1hr'
)
    
data = [candlesChart, opens, movingAverage1hrPlot]
fig = go.Figure(data=data)
py.plot(fig, filename=os.path.join(os.path.dirname(__file__), 'plots/eth_usd_candles.html'))
