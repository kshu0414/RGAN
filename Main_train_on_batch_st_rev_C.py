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
from data_generation import data_Gen, data_Dis
from utils import sine_wave, draw_train_loss, draw_generated_signal, draw_D_loss
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils.generic_utils import Progbar
import scipy.io
from termcolor import colored 
import keras.backend as K
import time
from keras.utils.generic_utils import Progbar
import matplotlib.pyplot as plt  


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def train_RGAN_batch(data_params,network_params,real_data,G_rounds,D_rounds,path,begin_epoch,end_epoch,G_lr,D_lr,decay,Pre_loss_file):
    start_time = time.time()
    nb_batch = int(np.ceil(data_params['data_size']/float(network_params['batch_size'])))
    batch_size = network_params['batch_size']

    Gen_loss = []
    Dis_loss = []

    Dreal_loss = []
    Dfake_loss = []
    Idx = []

    Training_log_file = 'training_log_'+str(begin_epoch)+'_'+str(end_epoch)+'.txt'
    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path,Training_log_file),'w') as filehandle:
        filehandle.write('Train on Batch Experiment: '+ path +' begin with epoch '+ str(begin_epoch)+' end with epoch '+ str(end_epoch)+'\n')
        filehandle.write('G_optimizer: adam D_optimizer: adam '+ 'G_lr: '+ str(G_lr) + 'D_lr: '+str(D_lr) + 'lr_decay every 100 epochs: '+ str(decay)+'\n')

    for epoch in range(begin_epoch,end_epoch+1):

        np.random.shuffle(real_data)
       # progress_bar = Progbar(target=nb_batch)
        curr_log = 'Epoch ' +str(epoch)+': ' + '\n'
        print(colored('Training epoch '+str(epoch),'yellow'))

        if epoch%100 == 0 and epoch!=1:
            G_lr = G_lr*decay
            D_lr = D_lr*decay
            K.set_value(Combined.optimizer.lr, G_lr)
            K.set_value(Discriminator.optimizer.lr, D_lr)
            curr_log = curr_log + 'G learning rate decayed to '+ str(G_lr) +'\n'
            curr_log = curr_log + 'D learning rate decayed to '+ str(D_lr) +'\n'            
        
        if epoch==begin_epoch or float(Dis_loss[-1]) > 0.7*float(Gen_loss[-1]):
            for index in range(0,nb_batch-D_rounds,D_rounds):
                for d in range(D_rounds):
                    batch_signal = real_data[(index+d)*batch_size:(index+d+1)*batch_size]

                    batch_real_label = np.ones((batch_size,data_params['data_len'],1))
                    batch_noise = np.random.normal(0,1,size=[batch_size,data_params['data_len'],network_params['latent_dim']])

                    batch_generated = Generator.predict(batch_noise)

                    batch_input = np.concatenate((batch_signal,batch_generated))
                    batch_label = np.concatenate((batch_real_label,np.zeros((batch_size,data_params['data_len'],1))))
                    d_b_loss = Discriminator.train_on_batch(batch_input,batch_label)
                    

                    print('The D loss at batch %s of epoch %s is %0.6f'%(index+d,epoch,d_b_loss))
                    
# =============================================================================
#                     dw = Combined.layers[2].get_weights()
#                     gw = Combined.layers[1].get_weights()
#                     print('the sample D weight is %0.9f-----g weight is %0.9f'%(dw[0][0][0],gw[0][0][0]))
# =============================================================================

                for g in range(G_rounds):
                    batch_noise = np.random.normal(0,1,size=[batch_size,data_params['data_len'],network_params['latent_dim']])
                    batch_trick = np.ones((batch_size,data_params['data_len'],1))

                    g_b_loss = Combined.train_on_batch(batch_noise,batch_trick)
                    

                    
                    print('The G loss at batch %s of epoch %s is %0.6f'%(index+d,epoch,g_b_loss))
                    
# =============================================================================
#                     dw = Combined.layers[2].get_weights()
#                     gw = Combined.layers[1].get_weights()
#                     print('the sample D weight is %0.9f-----g weight is %0.9f'%(dw[0][0][0],gw[0][0][0]))
# =============================================================================
                #progress_bar.update(index + 1)
                while g_b_loss> 1.01*d_b_loss:
                    
                    batch_noise = np.random.normal(0,1,size=[batch_size,data_params['data_len'],network_params['latent_dim']])
                    batch_trick = np.ones((batch_size,data_params['data_len'],1))

                    g_b_loss = Combined.train_on_batch(batch_noise,batch_trick)
                    
                    print('train extra epoch, the G loss at batch %s of epoch %s is %0.6f,target is %0.6f'%(index+d,epoch,g_b_loss,1.01*d_b_loss))
                    dw = Combined.layers[2].get_weights()
                    gw = Combined.layers[1].get_weights()
                    print('the sample D weight is %0.9f-----G weight is %0.9f'%(dw[0][0][0],gw[0][0][0]))
                
        else:
            curr_log = curr_log + '   D freezing because of the loss imbalance \n'
            for g in range(G_rounds):
                batch_noise = np.random.normal(0,1,size=[batch_size,data_params['data_len'],network_params['latent_dim']])
                batch_trick = np.ones((batch_size,data_params['data_len'],1))

                Combined.train_on_batch(batch_noise,batch_trick)
                
        input_noise = np.random.normal(0,1,size=(data_params['data_size'],data_params['data_len'],network_params['latent_dim']))
        generated_data = Generator.predict(input_noise)

        # Evaluate G loss and D loss
        Idx.append(epoch)
        input_trick = np.ones((data_params['data_size'],data_params['data_len'],1))
        Gen_loss.append(Combined.evaluate(input_noise,input_trick))
        curr_log = curr_log + '   Generator loss:' + str(Gen_loss[-1]) + '\n'

        input_signal = np.concatenate((real_data,generated_data))
        input_label = np.ones((2*data_params['data_size'],data_params['data_len'],1))
        input_label[data_params['data_size']:,:,:] = 0
        Dis_loss.append(Discriminator.evaluate(input_signal,input_label))
        curr_log = curr_log + '   Discriminator loss:' + str(Dis_loss[-1]) + '\n'
        print('now Gen_loss is %0.6f'%Gen_loss[-1])
        print('now Dis_loss is %0.6f'%Dis_loss[-1])
        
        # Evaluate D loss in real and fake dataset respectively
        Dreal_loss.append(Discriminator.evaluate(real_data,np.ones((data_params['data_size'],data_params['data_len'],1))))
        Dfake_loss.append(Discriminator.evaluate(generated_data,np.zeros((data_params['data_size'],data_params['data_len'],1))))

        # Draw loss curves after every 10 epochs
        if epoch%10 == 0:
            plotdata = {'eval_idx': Idx, 'Dreal_loss':Dreal_loss, 'Dfake_loss':Dfake_loss}
            draw_D_loss(plotdata,path,epoch)
            plotdata_GD = {'G_loss':Gen_loss,'D_loss':Dis_loss,'G_idx':Idx,'D_idx':Idx}
            draw_train_loss(plotdata_GD,path,epoch)
            curr_log = curr_log + 'Loss curve generated'+'\n'
		
        # Save model weights every 10 epochs
        if epoch%10 == 0:
            Gen_wfile = 'GW_epoch_'+str(epoch)+'.hdf5'
            Dis_wfile = 'DW_epoch_'+str(epoch)+'.hdf5'
            Generator.save_weights(os.path.join(path,Gen_wfile), True)
            Discriminator.save_weights(os.path.join(path,Dis_wfile), True)

		# Plot generated examples every 10 epochs
        if epoch%10 == 0:
            draw_generated_signal(generated_data,path,epoch=epoch)
            curr_log = curr_log + '   Generated example saved'+'\n'

		# Write log file
        with open(os.path.join(path,Training_log_file),'a') as filehandle:
            filehandle.write(curr_log)
            filehandle.write('Training time: '+ str((time.time()-start_time)/60) +' min \n')
            filehandle.write('\n')
            filehandle.close()
    
    scipy.io.savemat(os.path.join(path,'Training_history.mat'),{'G_loss':Gen_loss, 'D_loss':Dis_loss, 'index': Idx})





if __name__ == "__main__":
    

    data_params = {'data_len':30,'nb_channel':1,'data_size':28*5*100}
    network_params = {'hidden_unit':100,'latent_dim':5,'dropout_rate':0,'batch_size':28}
    D_lr = 0.001
    G_lr = 0.001
    decay = 0.707

    Generator = generator_model(data_params,network_params)
    print(Generator.summary())

    Discriminator = discriminator_model(data_params,network_params)
    Discriminator.compile(
        optimizer=SGD(lr=D_lr),
        loss=average_crossentropy,
    )
    print(Discriminator.summary())

    noise = Input(shape=(data_params['data_len'],network_params['latent_dim'],))
    generated = Generator(noise)
    fake = Discriminator(generated)
    Discriminator.trainable = False

    Combined = Model(inputs=noise,outputs=fake)
    Combined.compile(
        optimizer=Adam(lr=G_lr),
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
    D_round = 3
    G_round = 1

    curr_path = 'Cyclic_D%dG%d_TOB_SGDlr%0.5fSGD%0.5f_decay%0.5f'%(D_round,G_round,D_lr,G_lr,decay)
    #Discriminator.load_weights(os.path.join(curr_path,'DW.hdf5'))
    #Generator.load_weights(os.path.join(curr_path,'GW.hdf5'))
    #Combined.set_weights(Generator.get_weights())
    train_RGAN_batch(data_params,network_params,real_data=sine_data,G_rounds=G_round,D_rounds=D_round,path=curr_path,begin_epoch=1,end_epoch=1000,G_lr=G_lr,D_lr=D_lr,decay=decay,Pre_loss_file=None)
    
    
    real_data=sine_data
    path=curr_path
    begin_epoch=1
    end_epoch=1000
    
