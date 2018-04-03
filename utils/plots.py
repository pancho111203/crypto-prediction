# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import plotly.graph_objs as go

def candlesPlot(ticks):
    return go.Candlestick(x=ticks.index,
                       open=ticks.open,
                       high=ticks.high,
                       low=ticks.low,
                       close=ticks.close)

def movingAveragePlot(ticks, windowSize):
    data = ticks.close.rolling(windowSize, min_periods=windowSize).mean()
    return go.Scatter(
        x=data.index,
        y=data.values,
        name='movingAverage'
    )
    
                
def tendencyShapes(ticks):
    shapes = []
    
    start = ticks.iloc[0].name
    prevT = ticks.iloc[0]['tendency']
    for (date, row) in ticks[1:-1].iterrows():
        tendency = row['tendency']
        
        if prevT != tendency:
            end = date
            
            #paint it here
            if prevT == 1.0:
                shape = {
                    'type': 'rect',
                    'xref': 'x',
                    'yref': 'paper',
                    'x0': start,
                    'y0': 0,
                    'x1': end,
                    'y1': 1,
                    'fillcolor': '#d3d3d3',
                    'opacity': 0.2,
                    'line': {
                        'width': 0,
                    }
                }
                    
                shapes.append(shape)
            
            prevT = tendency
            start = date
    
    if prevT == 1.0:
        end = ticks.iloc[-1].name
        shape = {
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': start,
            'y0': 0,
            'x1': end,
            'y1': 1,
            'fillcolor': '#d3d3d3',
            'opacity': 0.2,
            'line': {
                'width': 0,
            }
        }
            
        shapes.append(shape)
    
    return shapes
    


def closesPlot(ticks):
    return go.Scatter(
        x=ticks.index,
        y=ticks.close,
        name='close'
    )


def volumePlot(ticks):
#    colors = []
#
#    for i in range(len(ticks['close'])):
#        if i != 0:
#            if df.Close[i] > df.Close[i-1]:
#                colors.append(INCREASING_COLOR)
#            else:
#                colors.append(DECREASING_COLOR)
#        else:
#            colors.append(DECREASING_COLOR)
#            
            
    return dict( 
        x=ticks.index,
        y=ticks.volume,
        type='bar',
        yaxis='y',
        name='Volume' 
    )


def bollingerBandsPlots(ticks, windowSize, std_mult=2):
    rolling_mean = ticks.close.rolling(window=windowSize, min_periods=windowSize).mean()
    rolling_std  = ticks.close.rolling(window=windowSize, min_periods=windowSize).std()
    upper_band = rolling_mean + (rolling_std*std_mult)
    lower_band = rolling_mean - (rolling_std*std_mult)
    plot1 = dict( x=ticks.index, y=upper_band, type='scatter', 
         line = dict( width = 1 ),
         marker=dict(color='#00ffff'), hoverinfo='none', 
         legendgroup='Bollinger Bands', name='Bollinger Bands')
                         
    plot2 = dict( x=ticks.index, y=lower_band, type='scatter',
         line = dict( width = 1 ),
         marker=dict(color='#00ffff'), hoverinfo='none',
         legendgroup='Bollinger Bands', showlegend=False )
         
    return (plot1, plot2)