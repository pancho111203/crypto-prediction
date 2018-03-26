# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import data_loader
import json
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd


model = 'model2.256lstmx2.stateful'
statsFile = 'stats_2018-03-15T18:33:35.111186.json'
dataFile = 'ETH-USD_60_2018-02-14T00:00:25+01:00_2018-03-14T00:00:25+01:00.json'

dataset = data_loader.getCandles('ETH-USD', 60, start='2018-02-14T00:00:25+01:00', end='2018-03-14T00:00:25+01:00', save=True)


batchA = dataset.iloc[0:6000, :]
batchB = dataset.iloc[10000:16000, :]
batchC = dataset.iloc[30000:36000, :]


def normalize(values):
    scaler = StandardScaler()
    res = scaler.fit_transform(values.reshape(-1, 1))
    return res[:, 0]

graphA = go.Scatter(
    x=list(range(0, len(batchA))),
    y=normalize(batchA.iloc[:, 3].values),
    name='A',
)

graphB = go.Scatter(
    x=list(range(0, len(batchB))),
    y=normalize(batchB.iloc[:, 3].values),
    name='B',
)

graphC = go.Scatter(
    x=list(range(0, len(batchC))),
    y=normalize(batchC.iloc[:, 3].values),
    name='C',
)


data = [graphA, graphB, graphC]
fig = go.Figure(data=data)
py.plot(fig, filename=os.path.join(os.path.dirname(__file__), '../plots/batch_std.html'))

