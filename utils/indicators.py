# -*- coding: utf-8 -*-
import pandas as pd


def fill_list_until(lst, until, value):
    toFill = until - len(lst) + 1
    for i in range(0, toFill):
        lst.append(value)



BULL = 1
BEAR = 0
def addTendency(candles, threshold=2):
    first = candles.iloc[0]['close']
    second = candles.iloc[1]['close']
    
    #initial state
    state = BULL if second >= first else BEAR
    states = [state, state]
    
    limit = second
    for i in range(2, len(candles)):
        curr = candles.iloc[i]['close']

        if state is BEAR:
            if curr < limit:
                limit = curr
                fill_list_until(states, i-1, BEAR)
                
            if curr > limit + threshold:
                fill_list_until(states, i-1, BULL)
                state = BULL
                limit = curr
        
        if state is BULL:
            if curr > limit:
                limit = curr
                fill_list_until(states, i-1, BULL)
                
            if curr < limit - threshold:
                fill_list_until(states, i-1, BEAR)
                state = BEAR
                limit = curr
                
    fill_list_until(states, len(candles) - 1, state)
    candles['tendency'] = pd.Series(states, index=candles.index)