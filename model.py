import os
import numpy as np
from keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


class Predict_model(object):

    def __init__(self, model_path='model1'): 
        model_from_path = os.path.join(os.path.dirname(__file__), 'prediction/checkpoint/{}/model.hdf5'.format(model_path))
        weights_path =  os.path.join(os.path.dirname(__file__), 'prediction/checkpoint/{}/weights.hdf5'.format(model_path))
        scaler_path = os.path.join(os.path.dirname(__file__), 'prediction/checkpoint/{}/scaler.pkl'.format(model_path))
        
        if os.path.isfile(model_from_path) and os.path.isfile(weights_path) and os.path.isfile(scaler_path):
            self.model = load_model(model_from_path)
            self.model.load_weights(weights_path)
            self.scaler = joblib.load(scaler_path)
        else:
            raise Exception("path dont exist")

    def get_model(self, x):

        x = self.scaler.transform(x) 
        x = np.array(x)
        x = np.reshape(x, (1, 1, 1))

        y = self.model.predict(x, batch_size=1)

        xt = self.scaler.inverse_transform(y)
        
        return xt[0,0]

    def buyer(self, x, xt):
        delta = 0.1
        if xt > x + delta:
            return 'buy'
        elif xt < x - delta:
            return 'sell'
        else:
            return 'stay'
