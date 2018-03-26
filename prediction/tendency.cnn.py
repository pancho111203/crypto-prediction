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
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils.dataloader import DataLoader
import argparse

import pandas as pd
from utils.indicators import addTendency
from torch.autograd import Variable
import data_loader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

checkpoint_name = 'tendency.cnn'
print('Starting...')
#X = random.sample(range(0, 1000), 1000)
#y = [X[k-2]+2 for (k, v) in enumerate(X)]

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
#dataset = data_loader.getCandles('ETH-USD', 60, start='2016-10-14T00:00:25+01:00', end='2018-03-22T00:00:25+01:00', save=True)
dataset = data_loader.getCandles('ETH-USD', 60, start='2018-02-14T00:00:25+01:00', end='2018-03-14T00:00:25+01:00', save=True)

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

x_features = 1 if len(trainX.shape) == 1 else trainX.shape[1]
look_back=6000
batch_size=100

# using unsqueeze because of this: https://github.com/pytorch/pytorch/issues/2539
trainX = torch.from_numpy(trainX).float()
trainY = torch.from_numpy(trainY).float().unsqueeze(1)
testX = torch.from_numpy(testX).float()
testY = torch.from_numpy(testY).float().unsqueeze(1)

if use_cuda:
    trainX, trainY, testX, testY = trainX.cuda(), trainY.cuda(), testX.cuda(), testY.cuda()

train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)

def collate_fn(_, indices, all_data):
    X = [all_data.data_tensor[index-look_back:index] for index in indices]
    y = [all_data.target_tensor[index] for index in indices]
    return (torch.stack(X), torch.stack(y))


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(list(range(look_back, len(train_dataset)))), collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(list(range(look_back, len(test_dataset)))), collate_fn=collate_fn)

checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}/weights.ckpt'.format(checkpoint_name))

# Model
class Model(nn.Module):
    def __init__(self, n_features, look_back):
        super(Model, self).__init__()
        self.n_features = n_features
        self.look_back = look_back

        self.avg_pool = nn.AvgPool1d(2)

        self.cnn_layer_1 = self.convlayer(5)
        self.cnn_layer_2 = self.convlayer(10)
        self.cnn_layer_3 = self.convlayer(50)
        self.cnn_layer_4 = self.convlayer(100)
        self.cnn_layer_5 = self.convlayer(1000)
        self.cnn_layer_6 = self.convlayer(3000, max_pool=False)

        self.conv_2 = nn.Sequential(
            nn.Conv1d(100, 200, 5, stride=2),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv1d(200, 200, 5, stride=2),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.drop1 = nn.Dropout(p=0.5)
        # TODO replace hardcoded 43000
        self.fc1 = nn.Linear(43000, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.drop3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 1)
    
    def convlayer(self, kernel_size, out_features=100, max_pool=True):
        #CNN
        #In:        (N, C(self.n_features), L(self.look_back))
        #Out:       (N, Cout(out_features), Lout)
        if max_pool:
            return nn.Sequential(
                nn.Conv1d(self.n_features, out_features, kernel_size, stride=2),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
        else:
            return nn.Sequential(
                nn.Conv1d(self.n_features, out_features, kernel_size, stride=2),
                nn.BatchNorm1d(out_features),
                nn.ReLU()
            )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.avg_pool(x)

        cnn_in = out.view(batch_size, self.n_features, -1)
        concat_cnns = torch.cat((self.cnn_layer_1(cnn_in), self.cnn_layer_2(cnn_in), self.cnn_layer_3(cnn_in), self.cnn_layer_4(cnn_in), self.cnn_layer_5(cnn_in), self.cnn_layer_6(cnn_in)), dim=2)

        out = self.conv_2(concat_cnns)
        out = self.conv_3(out)
        
        out = out.view(batch_size, -1)
        out = self.drop1(out)
        out = F.relu(self.bn1(self.fc1(out)))

        out = self.drop2(out)
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.drop3(out)
        out = F.sigmoid(self.fc3(out))
        return out

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
    net = Model(x_features, look_back)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

print('Model params: {}'.format(sum([param.nelement() for param in net.parameters()])))

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)


# Training
def train(epoch):
    net.train()
    optimizer.zero_grad()
    
    loss_total = 0
    for i, (X, y) in enumerate(train_dataloader):
        X, y = Variable(X), Variable(y)
        y_pred = net(X)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        loss_total += loss.data[0]

    print('TRAIN: Avg Loss: {}'.format(loss_total / len(train_dataloader)))

def test(epoch, save=True):
    global best_loss
    net.eval()

    loss_total = 0
    for i, (X, y) in enumerate(test_dataloader):
        X, y = Variable(X, volatile=True), Variable(y)
        y_pred = net(X)

        loss = criterion(y_pred, y)

        loss_total += loss.data[0]

    avg_loss = loss_total / len(test_dataloader)
    print('TEST: Avg Loss: {}'.format(avg_loss))

    if (avg_loss < best_loss or epoch % 10 == 0) and save:
        print('New best, Saving...')
        state = {
            'net': net.module if use_cuda else net,
            'loss': avg_loss,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path)
        best_loss = avg_loss

if not args.visualize_results:
    for epoch in range(start_epoch, start_epoch+3000):
        print('\nEpoch: %d' % epoch)
        start = time.time()
        train(epoch)
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

    
