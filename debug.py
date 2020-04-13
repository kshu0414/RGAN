import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import os.path
import math
import random
import tensorflow as tf
import numpy as np
from keras.models import Sequential
import statistics
from RGAN_model import generator_model, discriminator_model, average_crossentropy
from data_generation_new import data_Gen, data_Dis
from utils import sine_wave, draw_train_loss, draw_generated_signal
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils.generic_utils import Progbar
import scipy.io
from termcolor import colored 
import keras.backend as K
import time



# Check the function of data generator
def debug_discriminator_data_generator():

    Discriminator = discriminator_model(data_params,network_params)
    Discriminator.compile(
        optimizer=SGD(lr=learning_rate),
        loss=average_crossentropy,
    )
    print(Discriminator.summary())

    Dis_train_data = data_Dis(data_params,network_params,sine_data,generated_list=None)
    Discriminator.fit_generator(generator=Dis_train_data,verbose=1)

if __name__ == "__main__":
    
    data_params = {'data_len':30,'nb_channel':1,'data_size':28*5*100}
    network_params = {'hidden_unit':100,'latent_dim':5,'dropout_rate':0,'batch_size':28}
    nb_epoch = 100
    learning_rate = 0.0001
    adam_lr = 0.0001
    adam_beta_1 = 0.5

    file_name = 'Sine_data.mat'
    if os.path.isfile(file_name):
        sine_data = scipy.io.loadmat(file_name)['samples']
    else:
        sine_data = sine_wave()
        scipy.io.savemat(file_name,{'samples':sine_data})

    debug_path = 'debug_folder'
    if not os.path.exists(debug_path):
        os.mkdir(debug_path)

    #debug_discriminator_data_generator()

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
    """scipy.io.savemat(os.path.join(debug_path,'Initial_W.mat'),{'Initial_DW':Discriminator.get_weights(),'Initial_GW':Generator.get_weights(),'Initial_CW':Combined.get_weights()})

    Dis_train_data = data_Dis(sine_data,data_params,network_params,generated_list=None)
    Discriminator.fit_generator(generator=Dis_train_data,verbose=1)

    scipy.io.savemat(os.path.join(debug_path,'Dtrained_W.mat'),{'Dtrained_DW':Discriminator.get_weights(),'Dtrained_GW':Generator.get_weights(),'Dtrained_CW':Combined.get_weights()})

    Gen_train_data = data_Gen(sine_data,data_params,network_params)
    Combined.fit_generator(generator=Gen_train_data,verbose=1)

    scipy.io.savemat(os.path.join(debug_path,'Gtrained_W.mat'),{'Gtrained_DW':Discriminator.get_weights(),'Gtrained_GW':Generator.get_weights(),'Gtrained_CW':Combined.get_weights()})"""

    with open(os.path.join(debug_path,'debug_log.txt'),'w') as filehandle:
        filehandle.write('D loss component that correspond to G output data \n')
    
    nb_epoch = 50
    for d in range(nb_epoch):
        Gen_train_data = data_Gen(sine_data,data_params,network_params)
        Combined.fit_generator(generator=Gen_train_data,verbose=1)

        input_noise = np.random.uniform(0,1,(data_params['data_size'],data_params['data_len'],network_params['latent_dim']))
        generated_list = Generator.predict(input_noise)

        Dis_eval_data = data_Dis(data_params,network_params,real_list=None,generated_list=generated_list)
        test_loss = Discriminator.evaluate_generator(generator=Dis_eval_data)
        with open(os.path.join(debug_path,'debug_log.txt'),'a') as filehandle:
            filehandle.write(str(test_loss)+' ')




