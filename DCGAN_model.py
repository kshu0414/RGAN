import os
import sys
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LeakyReLU, Conv1D, Input, Flatten, Conv2DTranspose, Lambda
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras import backend as K

def build_discriminator(data_params,network_params):
    input_signal = Input(shape=(data_params['data_len'],data_params['nb_channel']))
    # Batch normalization?
    # First layer [64,1]->[16,64]
    fake = Conv1D(64,kernel_size=9,strides=4,padding='same')(input_signal)
    fake = LeakyReLU(alpha=0.2)(fake)
    # Phase shuffle?
    # First layer [16,64]->[4,128]
    fake = Conv1D(128,kernel_size=9,strides=4,padding='same')(fake)
    fake = LeakyReLU(alpha=0.2)(fake)
    fake = Flatten()(fake)
    fake = Dense(1,activation='sigmoid')(fake)
    Dis = Model(inputs=input_signal,outputs=fake)
    return(Dis)

def build_generator(data_params,network_params):
    input_noise = Input(shape=(network_params['latent_dim'],))
    # Batch normalization?
    # FC and reshape [30,]->[4,128]
    fake_signal = Dense(4*128,activation='relu')(input_noise)
    fake_signal = Reshape((-1,1,128))(fake_signal)
    # First layer [4,128]->[16,64]
    fake_signal = Conv2DTranspose(64,kernel_size=(9,1),strides=(4,1),padding='same',data_format='channels_last')(fake_signal)
    fake_signal = LeakyReLU(alpha=0.2)(fake_signal)
    # Second layer [16,64]->[64,1]
    fake_signal = Conv2DTranspose(1,kernel_size=(9,1),strides=(4,1),padding='same',data_format='channels_last')(fake_signal)
    fake_signal = LeakyReLU(alpha=0.2)(fake_signal)
    fake_signal = Lambda(K.squeeze,arguments={'axis':2})(fake_signal)
    fake_signal = Activation('tanh')(fake_signal)
    Gen = Model(inputs=input_noise,outputs=fake_signal)
    return(Gen)



if __name__ == "__main__":
    data_params = {'data_len':64,'nb_channel':1,'data_size':28*5*100}
    network_params = {'hidden_unit':100,'latent_dim':30,'dropout_rate':0,'batch_size':28}
    Generator = build_generator(data_params,network_params)
    #Discriminator = build_discriminator(data_params,network_params)

    print(Generator.summary())


#Single connected layer: https://stackoverflow.com/questions/56825036/make-a-non-fully-connected-singly-connected-neural-network-in-keras
        
