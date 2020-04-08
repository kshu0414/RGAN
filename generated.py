import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import os.path
import math
import random
import tensorflow as tf
import numpy as np
from keras.models import Sequential
import statistics
from RGAN_model import generator_model, discriminator_model, average_crossentropy
from data_generation import data_Gen, data_Dis
from utils import sine_wave, draw_train_loss
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils.generic_utils import Progbar
import scipy.io
from termcolor import colored 
import keras.backend as K

path_list  = {'D1G2_lr0.001','D1G2_lr0.002','D1G3_lr0.001','D1G3_lr0.002','D1G1_lr0.0001'}
data_params = {'data_len':30,'nb_channel':1,'data_size':28*5*100}
network_params = {'hidden_unit':100,'latent_dim':5,'dropout_rate':0,'batch_size':28}
Generator = generator_model(data_params,network_params)
Discriminator = discriminator_model(data_params,network_params)

for path in path_list:
    Generator.load_weights(os.path.join(path,'GW.hdf5'))
    Discriminator.load_weights(os.path.join(path,'DW.hdf5'))
    input_noise = np.random.uniform(0,1,(data_params['data_size'],data_params['data_len'],network_params['latent_dim']))
    generated_list = Generator.predict(input_noise)
    scipy.io.savemat(os.path.join(path,'generated.mat'),{'generated':generated_list})
