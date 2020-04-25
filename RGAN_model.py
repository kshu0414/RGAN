import os
import sys
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
import time
from keras.layers import Bidirectional, Conv1D, MaxPooling1D, Input, Flatten, Multiply, AveragePooling1D, Embedding, RepeatVector, Concatenate
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras import backend as K


# implementation of wasserstein loss
def average_crossentropy(y_true, y_pred):
    #loss = -K.mean(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred),axis=-1)
    loss = K.mean(K.maximum(y_pred,0) - y_true*y_pred + K.log(1+K.exp(-K.abs(y_pred))),axis=-1)
    return loss

def generator_model(data_params,network_params):
    input_noise = Input(shape=(data_params['data_len'],network_params['latent_dim'],))
    fake_signal = LSTM(network_params['hidden_unit'], activation='tanh',recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True)(input_noise)
    fake_signal = Dense(data_params['nb_channel'],activation=None,use_bias=True,kernel_initializer='TruncatedNormal',bias_initializer='TruncatedNormal')(fake_signal)
    Gen = Model(inputs=input_noise,outputs=fake_signal)
    return(Gen)

def discriminator_model(data_params,network_params):
    input_signal = Input(shape=(data_params['data_len'],data_params['nb_channel']))
    fake = LSTM(network_params['hidden_unit'], activation='tanh',recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
            return_sequences=True)(input_signal)
    fake = Dense(1,activation=None,use_bias=True,kernel_initializer='TruncatedNormal',bias_initializer='TruncatedNormal')(fake)
    Dis = Model(inputs=input_signal,outputs=fake)
    return(Dis)


#Single connected layer: https://stackoverflow.com/questions/56825036/make-a-non-fully-connected-singly-connected-neural-network-in-keras
        
