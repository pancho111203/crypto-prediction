# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools
import data_loader
from utils.indicators import addTendency
from utils.plots import candlesPlot, closesPlot, movingAveragePlot, bollingerBandsPlots, volumePlot, tendencyShapes

ticks = data_loader.getCandles('ETH-USD', 60, start='2016-10-14T00:00:25+01:00', end='2018-03-22T00:00:25+01:00', save=True)
#ticks = data_loader.getCandles('ETH-USD', 3600, save=False)
addTendency(ticks, threshold=0.10)


ticksSlice = ticks
tenShapes = tendencyShapes(ticksSlice)
layout = go.Layout(
    xaxis = dict(
            autorange=True
    ),
    yaxis = dict(
            autorange=True
    ),
    shapes = tenShapes
)
        
data = [closesPlot(ticksSlice)]
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename=os.path.join(os.path.dirname(__file__), '../plots/tests.html'))
