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
from utils import sine_wave, draw_train_loss, draw_generated_signal
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils.generic_utils import Progbar
import scipy.io
from termcolor import colored 
import keras.backend as K
import time

def train_RGAN(data_params,network_params,G_rounds,D_rounds,path):
    start_time = time.time()
    Dis_loss = []
    Gen_loss = []
    D_idx = []
    G_idx = []
    generated_matfile = 'generated_2.mat'
    loss_plotfile = 'training_loss_2.png'
    Gen_wfile = 'GW_2.hdf5'
    Dis_wfile = 'DW_2.hdf5'
    Training_log_file = 'training_log_2.txt'

    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path,Training_log_file),'w') as filehandle:
        filehandle.write('Experiment: '+ path +'\n')

    for i in range(100,100+nb_epoch):
        curr_log = 'Epoch ' +str(i+1)+': ' + '\n'
        print(colored('Training epoch '+str(i+1),'yellow'))
        input_noise = np.random.uniform(0,1,(data_params['data_size'],data_params['data_len'],network_params['latent_dim']))
        generated_list = Generator.predict(input_noise)
        if i%10 == 0:
            curr_log = curr_log + '   Generated example saved'+'\n'
            draw_generated_signal(generated_list,path,epoch=i)
        if i==100 or float(Dis_loss[-1][0]) > 0.7*float(Gen_loss[-1][0]):
            train_D = True
            for d in range(D_rounds):
                print(colored('Training discriminator '+str(d+1),'blue'))
                curr_log = curr_log + '   D trainig round '+str(d+1)+' loss:'
                Dis_train_data = data_Dis(sine_data,generated_list,data_params,network_params)
                Dis_train_his = Discriminator.fit_generator(generator=Dis_train_data,verbose=1)
                curr_log = curr_log + str(Dis_train_his.history['loss']) + '\n'
                Dis_loss.append(Dis_train_his.history['loss'])
                D_idx.append(i*(D_rounds+G_rounds)+d)
        else:
            train_D = False
            curr_log = curr_log + '   D freezing because of the loss imbalance'
            
        if np.isnan(Dis_train_his.history['loss']):
            break

        for g in range(G_rounds):
            print(colored('Training generator '+str(g+1),'red'))
            curr_log = curr_log + '   G trainig round '+str(g+1)+' loss:'
            Gen_train_data = data_Gen(sine_data,data_params,network_params)
            Gen_train_his = Combined.fit_generator(generator=Gen_train_data,verbose=1)
            curr_log = curr_log + str(Gen_train_his.history['loss']) + '\n'
            Gen_loss.append(Gen_train_his.history['loss'])
            if train_D:
                G_idx.append(D_idx[-1]+g)
            else:
                G_idx.append(G_idx[-1]+1)
        
        if np.isnan(Gen_train_his.history['loss']):
            curr_log = curr_log + 'Training interrupted'
            break
        generated_list = Generator.predict(input_noise)
        scipy.io.savemat(os.path.join(path,generated_matfile),{'generated'+str(i):generated_list})
        
        with open(os.path.join(path,Training_log_file),'a') as filehandle:
            filehandle.write(curr_log)
            filehandle.write('\n')
            filehandle.close()


    plotdata = {'G_loss':Gen_loss,'D_loss':Dis_loss,'G_idx':G_idx,'D_idx':D_idx}
    scipy.io.savemat(os.path.join(path,'Training_history.mat'),{'G_loss':Gen_loss, 'D_loss':Dis_loss})
    draw_train_loss(plotdata,os.path.join(path,loss_plotfile))
    Generator.save_weights(os.path.join(path,Gen_wfile), True)
    Discriminator.save_weights(os.path.join(path,Dis_wfile), True)

    with open(os.path.join(path,Training_log_file),'a') as filehandle:
            filehandle.write('Training time: '+ str((time.time()-start_time)/60) +' min')
            filehandle.close()

if __name__ == "__main__":
    
    data_params = {'data_len':30,'nb_channel':1,'data_size':28*5*100}
    network_params = {'hidden_unit':100,'latent_dim':5,'dropout_rate':0,'batch_size':28}
    nb_epoch = 100
    learning_rate = 0.0001
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
    
    #Combined.load_weights('Initial_GW.hdf5')
    #Discriminator.load_weights('Initial_DW.hdf5')
    #K.set_value(Combined.optimizer.lr, .001)
    #K.set_value(Discriminator.optimizer.lr, .001)
    #train_RGAN(data_params,network_params,G_rounds=2,D_rounds=1,path='D1G2_lr0.001')

    curr_path = 'D3G1_lr0.0001_loss_freeze'
    Discriminator.load_weights(os.path.join(curr_path,'DW.hdf5'))
    Generator.load_weights(os.path.join(curr_path,'GW.hdf5'))
    Combined.set_weights(Generator.get_weights())
    train_RGAN(data_params,network_params,G_rounds=1,D_rounds=3,path=curr_path)
