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

def train_RGAN(data_params,network_params,G_rounds,D_rounds,path):
    Dis_loss = []
    Gen_loss = []
    D_idx = []
    G_idx = []
    generated_matfile = 'generated.mat'
    loss_plotfile = 'training_loss.png'
    Gen_wfile = 'GW.hdf5'
    Dis_wfile = 'DW.hdf5'

    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(nb_epoch):
        print(colored('Training epoch '+str(i+1),'yellow'))
        input_noise = np.random.uniform(0,1,(data_params['data_size'],data_params['data_len'],network_params['latent_dim']))
        generated_list = Generator.predict(input_noise)
        for d in range(D_rounds):
            print(colored('Training discriminator '+str(d+1),'blue'))
            Dis_train_data = data_Dis(sine_data,generated_list,data_params,network_params)
            Dis_train_his = Discriminator.fit_generator(generator=Dis_train_data,verbose=0)
            Dis_loss.append(Dis_train_his.history['loss'])
            D_idx.append(i*(D_rounds+G_rounds)+d)
            
        if np.isnan(Dis_train_his.history['loss']):
            print('Training stopped at epoch '+str(i+1))
            break

        for g in range(G_rounds):
            print(colored('Training generator '+str(g+1),'red'))
            Gen_train_data = data_Gen(sine_data,data_params,network_params)
            Gen_train_his = Combined.fit_generator(generator=Gen_train_data,verbose=0)
            Gen_loss.append(Gen_train_his.history['loss'])
            G_idx.append(i*(D_rounds+G_rounds)+D_rounds+g)
        
        if np.isnan(Gen_train_his.history['loss']):
            print('Training stopped at epoch '+str(i+1))
            break
        generated_list = Generator.predict(input_noise)
        scipy.io.savemat(os.path.join(path,generated_matfile),{'generated'+str(i):generated_list})
    
    plotdata = {'G_loss':Gen_loss,'D_loss':Dis_loss,'G_idx':G_idx,'D_idx':D_idx}
    scipy.io.savemat(os.path.join(path,'Training_history.mat'),{'G_loss':Gen_loss, 'D_loss':Dis_loss})
    draw_train_loss(plotdata,os.path.join(path,loss_plotfile))
    Generator.save_weights(os.path.join(path,Gen_wfile), True)
    Discriminator.save_weights(os.path.join(path,Dis_wfile), True)

if __name__ == "__main__":
    
    data_params = {'data_len':30,'nb_channel':1,'data_size':28*5*100}
    network_params = {'hidden_unit':100,'latent_dim':5,'dropout_rate':0,'batch_size':28}
    nb_epoch = 100
    learning_rate = 0.001
    adam_lr = 0.0001
    adam_beta_1 = 0.5

    Generator = generator_model(data_params,network_params)
    print(Generator.summary())

    Discriminator = discriminator_model(data_params,network_params)
    Discriminator.compile(
        optimizer=SGD(lr=learning_rate),
        loss=average_crossentropy,
    )
    print(Discriminator.summary())

    noise = Input(shape=(data_params['data_len'],network_params['latent_dim'],))
    generated = Generator(noise)
    fake = Discriminator(generated)
    Discriminator.trainable = False

    Combined = Model(inputs=noise,outputs=fake)
    Combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=average_crossentropy,
    )
    print(Combined.summary())
    Discriminator.save_weights('Initial_DW.hdf5',True)
    Combined.save_weights('Initial_GW.hdf5',True)


    file_name = 'Sine_data.mat'
    if os.path.isfile(file_name):
        sine_data = scipy.io.loadmat(file_name)['samples']
    else:
        sine_data = sine_wave()
        scipy.io.savemat(file_name,{'samples':sine_data})
    
    #train_RGAN(data_params,network_params,G_rounds=2,D_rounds=1,path='D1G2_lr0.001')

    #Combined.load_weights('Initial_GW.hdf5')
    #Discriminator.load_weights('Initial_DW.hdf5')
    #K.set_value(Combined.optimizer.lr, .002)
    #K.set_value(Discriminator.optimizer.lr, .002)
    #train_RGAN(data_params,network_params,G_rounds=2,D_rounds=1,path='D1G2_lr0.002')

    Combined.load_weights('Initial_GW.hdf5')
    Discriminator.load_weights('Initial_DW.hdf5')
    K.set_value(Combined.optimizer.lr, .001)
    K.set_value(Discriminator.optimizer.lr, .001)
    train_RGAN(data_params,network_params,G_rounds=1,D_rounds=3,path='D3G1_lr0.001')

    Combined.load_weights('Initial_GW.hdf5')
    Discriminator.load_weights('Initial_DW.hdf5')
    train_RGAN(data_params,network_params,G_rounds=1,D_rounds=4,path='D4G1_lr0.001')
