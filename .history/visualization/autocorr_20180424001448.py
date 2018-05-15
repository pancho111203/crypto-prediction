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

candles = data_loader.getCandles('ETH-USD', 60, start='2018-02-14T00:00:25+01:00', end='2018-03-14T00:00:25+01:00', save=True)
data = candles['open']
ewma = pd.ewma(data, span=30)
residual = ewma - data

fig, axes = plt.subplots(nrows=3)
fig.tight_layout()

axes[0].plot(data)
pd.plotting.autocorrelation_plot(data, ax=axes[1])

axes[2].plot(residual)
plt.show()