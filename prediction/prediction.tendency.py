# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pandas as pd
import matplotlib.pyplot as plt
import data_loader
import numpy
import math
import argparse
import keras
import plotly.offline as py
import plotly.graph_objs as go
from keras.optimizers import Adam
import logging
from utils.indicators import addTendency

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

checkpoint_name = 'tendency.256lstmx2.stateful'

if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'checkpoint/{}'.format(checkpoint_name))):
    dire = os.path.join(os.path.dirname(__file__), 'checkpoint/')
    os.system("cd {}; mkdir {}".format(dire, checkpoint_name))

parser = argparse.ArgumentParser(description='Predict crypto prices')
parser.add_argument('-v', '--visualize', action='store_const',
                   const=True, default=False,
                   help='Only visualize results of previously saved model')
parser.add_argument('-r', '--resume', action='store_const',
                   const=True, default=False,
                   help='resume from previous checkpoint')

args = parser.parse_args()


"""Get Data"""
dataset = data_loader.getCandles('ETH-USD', 60, start='2018-03-12T13:19:54.527842', end='2018-03-15T13:19:54.527861', save=True)
addTendency(dataset, threshold=3)

scaler = StandardScaler()
scaler.fit(dataset[['open', 'volume']])
t = scaler.transform(dataset[['open', 'volume']])
tdf = pd.DataFrame(t, columns=['open_norm', 'vol_norm'], index=dataset.index)
dataset = pd.concat([dataset, tdf], axis=1)
"""###Split data into training and test. Training is the past, test is the future."""

# split into train and test sets
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset.iloc[0:train_size,:], dataset.iloc[train_size:len(dataset),:]

# print(len(train), len(test))

"""###Convert data into pairs: (features, targets)"""
trainX = train[['open_norm', 'vol_norm']].values
trainY = train['tendency'].values

testX = test[['open_norm', 'vol_norm']].values
testY = test['tendency'].values

"""###Reshape data to fit the LSTM expected format (samples, time_steps, features)"""
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

"""###Build a very simple LSTM with 4 nodes connected to a 1 neuron output layer:"""

# create and fit the LSTM network
model = Sequential()
# model.add(LSTM(256, input_shape=(1, 1), return_sequences=True))
# model.add(LSTM(256))
model.add(LSTM(256, batch_input_shape=(1, 1, 2), stateful=True, return_sequences=True))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(LSTM(256, stateful=True))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()


"""### Checkpointing """
checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}/weights.hdf5'.format(checkpoint_name))
model_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}/model.hdf5'.format(checkpoint_name))
checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error', 'MAE'])

if args.resume and os.path.isfile(checkpoint_path) and os.path.isfile(model_path):
    model = load_model(model_path)
    model.load_weights(checkpoint_path)
    print('Loaded weigths from checkpoint: {}'.format(checkpoint_path))
else:
    print('No checkpoint found.')
    model.save(model_path)


"""###Train the model."""

if not args.visualize:
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=1, validation_data=(testX, testY), callbacks=[checkpointer])

"""###Now check the predicted values for training and test data"""

# make predictions
trainPredict = model.predict(trainX, batch_size=1)
testPredict = model.predict(testX, batch_size=1)
print(trainPredict.shape)
print(testPredict.shape)


"""Training Graphs"""
original = go.Scatter(
    x=dataset.index,
    y=dataset['close'],
    name='Original Data'
)

trainP = go.Scatter(
    x=train.index,
    y=trainPredict[:, 0],
    name='Train Predict Data',
    yaxis='y2',
    fill='tozeroy',
    line=dict(
        width=0
    )
)

testP = go.Scatter(
    x=test.index,
    y=testPredict[:, 0],
    name='Test Predict Data',
    yaxis='y2',
    fill='tozeroy',
    line=dict(
        width=0
    )
)

layout = go.Layout(
    yaxis=dict(
        title='Price'
    ),
    yaxis2=dict(
        title='state',
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

data = [original, trainP, testP]
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename=os.path.join(os.path.dirname(__file__), '../plots/{}.html'.format(checkpoint_name)))
