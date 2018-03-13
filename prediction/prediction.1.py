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

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

checkpoint_name = 'model2.256lstmx2'

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
dataset = data_loader.getCandles('ETH-USD', 60, '2018-02-08T00:00:25+01:00', '2018-03-08T00:00:25+01:00')[['open']]

"""###Normalize data"""

# normalize the dataset
dataset1 = dataset.values[:-1]
dataset2 = dataset.values[1:]

dataset = dataset1/dataset2
"""###Split data into training and test. Training is the past, test is the future."""

# split into train and test sets
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# print(len(train), len(test))

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

print("Train y")
print(trainY.shape)
print(testY.shape)
# print(testY)
"""###Reshape data to fit the LSTM expected format (samples, time_steps, features)"""
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print("Train X")
print(trainX.shape)
print(testX.shape)

"""###Build a very simple LSTM with 4 nodes connected to a 1 neuron output layer:"""

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(256, input_shape=(1, look_back), return_sequences=True))
model.add(LSTM(256))
# model.add(LSTM(4, batch_input_shape=(1, 1, look_back), stateful=True))
model.add(Dense(1))

model.summary()


"""### Checkpointing """
checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}/weights.hdf5'.format(checkpoint_name))
model_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}/model.hdf5'.format(checkpoint_name))
checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['MSE', 'MAE'])

if args.resume and os.path.isfile(checkpoint_path) and os.path.isfile(model_path):
    model = load_model(model_path)
    model.load_weights(checkpoint_path)
    print('Loaded weigths from checkpoint: {}'.format(checkpoint_path))
else:
    print('No checkpoint found.')
    model.save(model_path)

"""###Train the model."""


if not args.visualize:
    model.fit(trainX, trainY, epochs=25, batch_size=1, verbose=1, validation_data=(testX, testY), callbacks=[checkpointer])

"""###Now check the predicted values for training and test data"""

# make predictions
trainPredict = model.predict(trainX, batch_size=1)
testPredict = model.predict(testX, batch_size=1)
print(trainPredict.shape)
print(testPredict.shape)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: {} % RMSE'.format(trainScore*100))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: {} % RMSE'.format(testScore*100))


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



"""Training Graphs"""
data = go.Scatter(
    x=pandas.DataFrame(dataset).index,
    y=pandas.DataFrame(dataset)[0],
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
