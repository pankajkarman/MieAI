import json
# from mei import Size_Distribution_Optics
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Activation, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TerminateOnNaN

def normalize(df):
    return (df - df.mean()) / df.std()

def normalize1(df):    
    mx = df.max()
    mn = df.min()    
    new = (df - mn) / (mx - mn)    
    return new

def load_opt(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)
    return data

# Recommended: Glorot_initializer for tanh and he_uniform for relu

# Keras default is glorot_uniform (also called xavier uniform initialiser)

def get_weight_init(activation):
    if activation=='relu':
        act = 'he_uniform'
    else:
        act = 'glorot_uniform'  
    return act

def create_model(
    input_size,
    output_size,
    hidden=[14, 16, 5, 10],
    dropout=[0.1, 0.3, 0.3, 0],
    activation=["relu", "tanh", "tanh", "tanh"],
    loss="mean_absolute_error",
    optimizer="adam",
    metrics=["mean_squared_error"],
):
    input_shape = (input_size,)
    print(f"Input shape: {input_shape}")

    # Create the model
    act1 = get_weight_init(activation[0])
    act2 = keras.regularizers.l1_l2(0.001)
    model = Sequential()
    model.add(
        Dense(
            hidden[0],
            input_shape=input_shape,
            activation=activation[0],
            kernel_initializer=act1,
            name="hidden_layer_1",
            use_bias=True,
        ),
    )
    model.add(Dropout(dropout[0]))
    model.add(BatchNormalization())

    if len(hidden) > 1:
        for i, (ly , act, drp) in enumerate(zip(hidden[1:], activation[1:], dropout[1:])):
            act1 = get_weight_init(act)
            model.add(
                Dense(
                    ly,
                    activation=act,
                    kernel_initializer=act1,
                    name="hidden_layer_%s" % (i + 2),
                    use_bias=True,
                )
            )
            model.add(Dropout(drp))
            model.add(BatchNormalization())
    model.add(Dense(output_size, name="output_layer"))

    # Configure the model and start training
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())
    return model