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
        ax.annotate(string, (1, 1), xytext=(-8, -8), ha='right', va='top',
                    size=14, xycoords='axes fraction', textcoords='offset points')

    fig, axes = plt.subplots(nrows=4, figsize=(8, 12))
    fig.tight_layout()

    axes[0].plot(data)
    label(axes[0], 'Raw Data')

    axes[1].acorr(data, maxlags=data.size-1)
    label(axes[1], 'Matplotlib Autocorrelation')

    tsaplots.plot_acf(data, axes[2])
    label(axes[2], 'Statsmodels Autocorrelation')

    pd.plotting.autocorrelation_plot(data, ax=axes[3])
    label(axes[3], 'Pandas Autocorrelation')

    # Remove some of the titles and labels that were automatically added
    for ax in axes.flat:
        ax.set(title='', xlabel='')
    plt.show()

candles = data_loader.getCandles('ETH-USD', 60, '2018-02-01T00:00:25+01:00', '2018-02-28T23:58:25+01:00')
data = candles['open']
stats_graph(data)