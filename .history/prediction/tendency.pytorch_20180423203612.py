# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse

import pandas as pd
from utils.indicators import addTendency
from torch.autograd import Variable
import data_loader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

checkpoint_name = 'tendency.pytorch'

if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'checkpoint/{}'.format(checkpoint_name))):
    dire = os.path.join(os.path.dirname(__file__), 'checkpoint/')
    os.system("cd {}; mkdir {}".format(dire, checkpoint_name))

parser = argparse.ArgumentParser(description='LSTM Crypto Prediction')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--visualize_results', '-v', action='store_true', default=False, help='use this to visualize the results of the currently trained model')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_loss = 99999

###Load data
dataset = data_loader.getCandles('ETH-USD', 60, start='2018-03-12T13:19:54.527842', end='2018-03-15T13:19:54.527861', save=True)
addTendency(dataset, threshold=0.05)

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

#TEST DATA
#import random
#trainX = np.random.random(10000)
#trainY = np.array([trainX[k-1]*2 for (k, v) in enumerate(trainX)])
#
#testX = np.random.random(1000)
#testY = np.array([testX[k-1]*2 for (k, v) in enumerate(testX)])



# reshape input to be [batch_size, seq_length, features]
x_features = 1 if len(trainX.shape) == 1 else trainX.shape[1]
batch_size = 21 # TRAIN DATA NEEDS TO BE DIVISIBLE BY THIS

trainX = np.reshape(trainX, (batch_size, int(trainX.shape[0]/batch_size), x_features))
trainY = np.reshape(trainY, (batch_size, int(trainY.shape[0]/batch_size), 1))
testX = np.reshape(testX, (1, testX.shape[0], x_features))
testY = np.reshape(testY, (1, testY.shape[0], 1))

checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}/weights.ckpt'.format(checkpoint_name))

# Model
class Model(nn.Module):
    def __init__(self, input_size, num_layers=2, hidden_size=256):
        super(Model, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.2, batch_first=True)
        self.dense = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if hidden is None:
            h0 = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size))
            c0 = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size))
        else:
            (h0, c0) = hidden
        
        output, hidden = self.lstm(x, (h0, c0))
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = F.sigmoid(self.dense(output))
        return output, hidden

if args.resume and os.path.isfile(checkpoint_path):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(checkpoint_path)
    net = checkpoint['net']
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    print('Loaded loss: {}, Starting epoch: {}'.format(best_loss, start_epoch))
else:
    print('==> Building model..')
    net = Model(x_features)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    net.train()

    X = torch.from_numpy(trainX).float()
    y = torch.from_numpy(trainY).float()
    
    if use_cuda:
        X, y = X.cuda(), y.cuda()

    optimizer.zero_grad()
    X, y = Variable(X), Variable(y)
    
    train_loss_total = 0
    hidden = None
    for i in range(0, len(X)):
        X_batch = X[i].view(1, X[i].shape[0], X[i].shape[1])
        y_batch = y[i].view(1, y[i].shape[0], y[i].shape[1])

        y_pred, hidden = net(X_batch, hidden)

        loss = criterion(y_pred, y_batch)
        loss.backward(retain_graph=False)
        optimizer.step()

        train_loss_total += loss.data[0]
        print('Batch {}: loss: {}'.format(i, loss.data[0]))

        hidden = (hidden[0].detach(), hidden[1].detach())

    print('TRAIN: Avg Loss: {}'.format(train_loss_total / len(X)))
    return y_pred

def test(epoch, save=True):
    global best_loss
    net.eval()

    X = torch.from_numpy(testX).float()
    y = torch.from_numpy(testY).float()
    
    if use_cuda:
        X, y = X.cuda(), y.cuda()

    X, y = Variable(X, volatile=True), Variable(y)
    y_pred, hidden = net(X)

    loss = criterion(y_pred, y)
    
    train_loss = loss.data[0]
    print('TEST: Avg Loss: {}'.format(train_loss))
    if train_loss < best_loss and save:
        print('New best, Saving...')
        state = {
            'net': net.module if use_cuda else net,
            'loss': train_loss,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path)
        best_loss = train_loss
        
    return y_pred

if not args.visualize_results:
    for epoch in range(start_epoch, start_epoch+3000):
        print('\nEpoch: %d' % epoch)
        start = time.time()
        y_pred = train(epoch)
        test(epoch)
        end = time.time()
        timeTaken = end - start
        print('Epoch {} took {} seconds'.format(epoch, timeTaken))
else:
    pass
    # import plotly.offline as py
    # import plotly.graph_objs as go
    
    # train_values = train(start_epoch).data.numpy()
    # test_values = test(start_epoch, save=False).data.numpy()

    # train_real_chart = go.Scatter(
    #     x=list(range(0, len(train_values))),
    #     y=trainY,
    #     name='train_real'
    # )

    # train_predicted_chart = go.Scatter(
    #     x=list(range(0, len(train_values))),
    #     y=train_values,
    #     name='train_predicted'
    # )

    # test_real_chart = go.Scatter(
    #     x=list(range(len(train_values), len(train_values) + len(test_values))),
    #     y=testY,
    #     name='test_real'
    # )

    # test_predicted_chart = go.Scatter(
    #     x=list(range(len(train_values), len(train_values) + len(test_values))),
    #     y=test_values,
    #     name='test_predicted'
    # )

    # data = [train_real_chart, train_predicted_chart, test_real_chart, test_predicted_chart]
    # fig = go.Figure(data=data)
    # py.plot(fig, filename=os.path.join(os.path.dirname(__file__), '../plots/prediction.html'))

    