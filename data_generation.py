from keras.utils import Sequence
import numpy as np
import random
from keras.models import Model, load_model
from keras.models import Sequential
import tensorflow as tf
from termcolor import colored


class data_Gen(Sequence):
    def __init__(self,real_list,data_params,network_params,shuffle=True):
        self._real_list = real_list
        self._nb_channels = data_params['nb_channel']
        self._batch_size = network_params['batch_size']
        self._shuffle = shuffle
        self._custom_idxs = []
        self._hidden_unit = network_params['hidden_unit']
        self._data_len = data_params['data_len']
        self._latent_dim = network_params['latent_dim']
        self._data_size = self._real_list.shape[0]
        self._custom_idxs = np.arange(self._data_size)+1
        if self._shuffle:
            random.shuffle(self._custom_idxs) 
        self._nb_batches = int(np.ceil(self._data_size/float(self._batch_size)))   
    
    def __len__(self):
        return self._nb_batches

    def on_epoch_end(self):
        if self._shuffle:
            random.shuffle(self._custom_idxs)

    def __getitem__(self,index):
        X,y = self.__data_generation(index)
        return X,y
    
    def __data_generation(self,index):
        #batch_label = np.zeros((self._batch_size))
        batch_fake = np.ones((self._batch_size,self._data_len,1))
        batch_noise = np.float32(np.random.normal(0, 1, (self._batch_size, self._data_len, self._latent_dim)))
        return batch_noise,batch_fake

class data_Dis(Sequence):
    def __init__(self,real_list,data_params,network_params,generated_list=None,shuffle=True):
        self._real_list = real_list
        self._generated_list = generated_list
        self._nb_channels = data_params['nb_channel']
        self._batch_size = network_params['batch_size']
        self._shuffle = shuffle
        self._custom_idxs = []
        self._hidden_unit = network_params['hidden_unit']
        self._data_len = data_params['data_len']
        self._data_size = self._real_list.shape[0]
        self._only_real = False
        if self._generated_list == None:
            self._only_real = True
        else:
            if self._data_size != self._generated_list.shape[0] and self._generated_list.shape[0] != 0:
                raise ValueError('Size of generated list different to that of real data list')
        self._custom_idxs = np.arange(self._data_size)
        if self._shuffle:
            random.shuffle(self._custom_idxs) 
        self._nb_batches = int(np.ceil(self._data_size/float(self._batch_size)))
        if self._only_real:
            print(colored('Only real data feed into discriminator','green'))
        else:
            print(colored('Both generated and real data feed into discriminator','green')) 
    
    def __len__(self):
        return self._nb_batches

    def on_epoch_end(self):
        if self._shuffle:
            random.shuffle(self._custom_idxs)

    def __getitem__(self,index):
        X,y = self.__data_generation(index)
        return X,y
    
    def __data_generation(self,index):
        batch_idxs = np.sort(self._custom_idxs[(index*self._batch_size):((index+1)*self._batch_size)])
        batch_input = np.array(self._real_list[batch_idxs,:,:])
        batch_label = np.ones((self._batch_size,self._data_len,1))

        if not self._only_real:
            batch_generated = np.array(self._generated_list[batch_idxs,:,:])
            batch_input = np.concatenate((batch_input,batch_generated),axis=0)
            batch_label = np.concatenate((batch_label,np.zeros((self._batch_size,self._data_len,1))),axis=0)

        return batch_input,batch_label
