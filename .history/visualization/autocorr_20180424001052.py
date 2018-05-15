# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
import matplotlib.pyplot as plt
import data_loader
import pandas as pd
import logging
from statsmodels.graphics import tsaplots

logging.basicConfig(level=logging.INFO)

def stats_graph(data):
    def label(ax, string):
        ax.annotate(string, (1, 1), xytext=(-8, -8), ha='right', va='top', xycoords='axes fraction', textcoords='offset points')

    fig, axes = plt.subplots(nrows=2)
    fig.tight_layout()

    axes[0].plot(data)
    label(axes[0], 'Raw Data')

    pd.plotting.autocorrelation_plot(data, ax=axes[1])
    label(axes[1], 'Pandas Autocorrelation')

    # Remove some of the titles and labels that were automatically added
    for ax in axes.flat:
        ax.set(title='', xlabel='')
    plt.show()

candles = data_loader.getCandles('ETH-USD', 60, start='2018-02-14T00:00:25+01:00', end='2018-03-14T00:00:25+01:00', save=True)
data = candles['open']
ewma = pd.ewma(data, span=30)

fig, axes = plt.subplots(nrows=1)
fig.tight_layout()

axes[0]
stats_graph(data)