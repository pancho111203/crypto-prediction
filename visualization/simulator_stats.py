# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import json
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd


model = 'model2.256lstmx2.stateful'
statsFile = 'stats_2018-03-15T18:33:35.111186.json'
dataFile = 'ETH-USD_60_2018-02-14T00:00:25+01:00_2018-03-14T00:00:25+01:00.json'

with open(os.path.join(os.path.dirname(__file__), '../prediction/checkpoint/{}/{}'.format(model, statsFile)), 'r') as f:
    stats = pd.DataFrame(json.load(f))


statsC= go.Scatter(
    x=stats[0],
    y=stats[1],
    name='stats',
    yaxis='y'
)

with open(os.path.join(os.path.dirname(__file__), '../data/candles/{}'.format(dataFile)), 'r') as f:
    candles = pd.DataFrame(json.load(f))
    
    
    
candlesC = go.Scatter(
    x=pd.to_datetime(candles[0], unit='s'),
    y=candles[4],
    name='close_price',
    yaxis='y2'
)



layout = go.Layout(
    yaxis=dict(
        title='Value'
    ),
    yaxis2=dict(
        title='Price',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        anchor='x',
        overlaying='y',
        side='right'
    )
)

data = [statsC, candlesC]
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename=os.path.join(os.path.dirname(__file__), '../plots/model2.256lstmx2.stateful_stats_2018-03-15T13:34:42.335998.html'))
