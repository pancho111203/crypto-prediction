import os
import numpy as np
from keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


def get_model(x, model_path):

    if os.path.isdir(model_path):
    
        model_form_path = os.path.join(os.path.dirname(__file__), '{}/model.hdf5'.format(model_path))
        weights_path =  os.path.join(os.path.dirname(__file__), '{}/weights.hdf5'.format(model_path))
        scaler_path = os.path.join(os.path.dirname(__file__), '{}/scaler.pkl'.format(model_path))
        
        model = load_model(model_form_path)
        model.summary()
        model.load_weights(weights_path)
        scaler = joblib.load(scaler_path)

        x = scaler.transform(x) 
        x = np.array(x)
        x = np.reshape(x, (1, 1, 1))

        y = model.predict(x, batch_size=1)

        xt = scaler.inverse_transform(y)

    else:
        print("path dont exist")
        xt = None
    
    return xt

def buyer(x, xt):
    delta = 0.1
    if xt > x + delta:
        return 'buy'
    elif xt < x - delta:
        return 'sell'
    else:
        return 'stay'

print(get_model(749.43,'prediction/checkpoint/'))