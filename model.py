import os
import tensorflow as tf
import numpy as np
from keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.optimizers import Adam


class Predict_model(object):

    def __init__(self, model_path='model1'): 
        model_from_path = os.path.join(os.path.dirname(__file__), 'prediction/checkpoint/{}/model.hdf5'.format(model_path))
        weights_path =  os.path.join(os.path.dirname(__file__), 'prediction/checkpoint/{}/weights.hdf5'.format(model_path))
        scaler_path = os.path.join(os.path.dirname(__file__), 'prediction/checkpoint/{}/scaler.pkl'.format(model_path))
        
        if os.path.isfile(model_from_path) and os.path.isfile(weights_path) and os.path.isfile(scaler_path):
            self.model = load_model(model_from_path)
            self.model.load_weights(weights_path)
            self.scaler = joblib.load(scaler_path)
            self.adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
            self.model.compile(loss='mean_squared_error', optimizer=self.adam)
            self.graph = tf.get_default_graph()
        else:
            raise Exception("path dont exist")

    def get_model(self, x):

        x = self.scaler.transform(x) 
        x = np.array(x)
        x = np.reshape(x, (1, 1, 1))

        with self.graph.as_default():
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
    
    def training(self, x, xt):
        x = self.scaler.transform(x) 
        x = np.array(x)
        trainX = np.reshape(x, (1, 1, 1))

        xt = self.scaler.transform(xt) 
        trainY = np.array(xt)

        with self.graph.as_default():
            self.model.fit(trainX, trainY, epochs=1, batch_size=1)

if __name__ == '__main__':
    predictor = Predict_model("model1")
    print(predictor.get_model(702))
    predictor.training(702, 700)
    print(predictor.get_model(702))