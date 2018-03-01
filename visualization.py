# -*- coding: utf-8 -*-
import json
import pandas as pd
import os
import plotly.offline as py
import plotly.graph_objs as go
import data_loader

candles = data_loader.getCandles('ETH-USD', 60, '2018-02-25T00:00:25+01:00', '2018-02-25T23:58:25+01:00')
# 1 hr moving average
movingAverage1hr = candles.open.rolling(60, min_periods=60).mean()

# Expo weighted moving average
ewma = pd.ewma(candles['open'], span=60)


ewmChart = go.Scatter(
    x=ewma.index,
    y=ewma.values,
    name='ewma_1hr'
)

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
    name='movingAverage_1hr'
)
    
data = [candlesChart, opens, movingAverage1hrPlot, ewmChart]
fig = go.Figure(data=data)
py.plot(fig, filename=os.path.join(os.path.dirname(__file__), 'plots/eth_usd_candles.html'))
