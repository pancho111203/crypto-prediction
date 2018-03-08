# -*- coding: utf-8 -*-
import logging
import os
logger = logging.getLogger(__name__)

import gdax

wsClient = gdax.WebsocketClient(url="wss://ws-feed.gdax.com", products="BTC-USD", channel)

def on_message(msg):
    print(msg)

wsClient.on_message = on_message
wsClient.start()
#wsClient.close()

