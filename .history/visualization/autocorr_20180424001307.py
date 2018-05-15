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

fig, axes = plt.subplots(nrows=2)
fig.tight_layout()

axes[0].plot(data)
axes[0].annotate('Data', (1,1))
pd.plotting.autocorrelation_plot(data, ax=axes[1])
axes[1].annotate('Autocorrelation', (1,1))

plt.show()