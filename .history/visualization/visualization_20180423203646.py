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

ticks = data_loader.getCandles('ETH-USD', 3600, save=False)
addTendency(ticks, threshold=0.1)


fig = tools.make_subplots(rows=2, cols=1)

fig.append_trace(candlesPlot(ticks), 1, 1)
fig.append_trace(closesPlot(ticks), 1, 1)
fig.append_trace(movingAveragePlot(ticks, 10), 1, 1)

(bbPlot1, bbPlot2) = bollingerBandsPlots(ticks, 10)
fig.append_trace(bbPlot1, 1, 1)
fig.append_trace(bbPlot2, 1, 1)

fig.append_trace(volumePlot(ticks), 2, 1)

fig['layout'].update(shapes = tendencyShapes(ticks))

py.plot(fig, filename=os.path.join(os.path.dirname(__file__), '../plots/visualization.html'))
