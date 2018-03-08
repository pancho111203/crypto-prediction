# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pandas
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

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

checkpoint_name = 'stateless'

parser = argparse.ArgumentParser(description='Predict crypto prices')
parser.add_argument('-v', '--visualize', action='store_const',
                   const=True, default=False,
                   help='Only visualize results of previously saved model')

args = parser.parse_args()


"""Get Data"""
dataset = data_loader.getCandles('ETH-USD', 60, '2018-01-25T00:00:25+01:00', '2018-02-25T00:00:25+01:00')[['open']]

"""###Normalize data"""

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

"""###Split data into training and test. Training is the past, test is the future."""

# split into train and test sets
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

"""###Convert data into pairs: (features, targets)"""

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

"""###Reshape data to fit the LSTM expected format (samples, time_steps, features)"""
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

"""###Build a very simple LSTM with 4 nodes connected to a 1 neuron output layer:"""

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
#model.add(LSTM(4, batch_input_shape=(1, 1, look_back), stateful=True))
model.add(Dense(1))

"""### Checkpointing """
checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}.hdf5'.format(checkpoint_name))
checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

if os.path.isfile(checkpoint_path):
    model.load_weights(checkpoint_path)
    print('Loaded weigths from checkpoint: {}'.format(checkpoint_path))
else:
    print('No checkpoint found.')

"""###Define the loss and optimizer. Train the model."""
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam)

if not args.visualize:
    model.fit(trainX, trainY, epochs=25, batch_size=1, verbose=1, validation_data=(testX, testY), callbacks=[checkpointer])

"""###Now check the predicted values for training and test data"""

# make predictions
trainPredict = model.predict(trainX, batch_size=1)
testPredict = model.predict(testX, batch_size=1)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
trainPredictPlot[:len(trainPredict), :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
testPredictPlot[len(trainPredict)+2:len(trainPredict)+len(testPredict)+2, :] = testPredict

# plot baseline and predictions

# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()

data = go.Scatter(
    x=pandas.DataFrame(scaler.inverse_transform(dataset)).index,
    y=pandas.DataFrame(scaler.inverse_transform(dataset))[0],
    name='Original Data'
)

train = go.Scatter(
    x=pandas.DataFrame(trainPredictPlot).index,
    y=pandas.DataFrame(trainPredictPlot)[0],
    name='Train Predict Data'
)

test = go.Scatter(
    x=pandas.DataFrame(testPredictPlot).index,
    y=pandas.DataFrame(testPredictPlot)[0],
    name='Test Predict Data'
)

data_plot = [data, train, test]
fig = go.Figure(data=data_plot)
py.plot(fig, filename=os.path.join(os.path.dirname(__file__), '../plots/{}.html'.format(checkpoint_name)))
