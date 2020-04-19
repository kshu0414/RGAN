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


def train_RGAN(data_params,network_params,G_rounds,D_rounds,path,begin_epoch,end_epoch,G_lr,D_lr,decay,Pre_loss_file):
    start_time = time.time()
    Dreal_loss = []
    Dfake_loss = []
    D_eval_idx = []

    if Pre_loss_file is None:
        Dis_loss = []
        Gen_loss = []
        D_idx = []
        G_idx = []
    else:
        Pre_loss = scipy.io.loadmat(os.path.join(path,Pre_loss_file))
        Dis_loss = Pre_loss['D_loss']
        Gen_loss = Pre_loss['G_loss']
        if D_rounds > G_rounds:
            G_idx = (D_rounds+G_rounds)*np.arange(1,begin_epoch)
            D_idx = np.arange(1,(D_rounds+G_rounds)*(begin_epoch-1))
            D_idx = np.delete(D_idx,G_idx)
        else:
            D_idx = (D_rounds+G_rounds)*np.arange(1,begin_epoch)
            G_idx = np.arange(1,(D_rounds+G_rounds)*(begin_epoch-1))
            G_idx = np.delete(G_idx,D_idx)
        Dis_loss = Dis_loss.tolist()
        Gen_loss = Gen_loss.tolist()
        G_idx = G_idx.tolist()
        D_idx = D_idx.tolist()
    
    Training_log_file = 'training_log_'+str(begin_epoch)+'_'+str(end_epoch)+'.txt'

    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path,Training_log_file),'w') as filehandle:
        filehandle.write('Experiment: '+ path +' begin with epoch '+ str(begin_epoch)+' end with epoch '+ str(end_epoch)+'\n')
        filehandle.write('G_optimizer: SGD D_optimizer: SGD '+ 'G_lr: '+ str(G_lr) + 'D_lr: '+str(D_lr) + 'lr_decay every 100 epochs: '+ str(decay)+'\n')

    # Training loop
    for i in range(begin_epoch,end_epoch+1):
        
        curr_log = 'Epoch ' +str(i)+': ' + '\n'
        print(colored('Training epoch '+str(i),'yellow'))
        input_noise = np.random.uniform(0,1,(data_params['data_size'],data_params['data_len'],network_params['latent_dim']))
        generated_list = Generator.predict(input_noise)
        
        # Adjust learning rate
        if i%100 == 0:
            G_lr = G_lr*decay
            D_lr = D_lr*decay
            K.set_value(Combined.optimizer.lr, G_lr)
            K.set_value(Discriminator.optimizer.lr, D_lr)
            curr_log = curr_log + 'G learning rate decayed to '+ str(G_lr) +'\n'
            curr_log = curr_log + 'D learning rate decayed to '+ str(D_lr) +'\n'

        ##### D training rounds
        if i==begin_epoch or float(Dis_loss[-1][0]) > 0.7*float(Gen_loss[-1][0]):
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
            curr_log = curr_log + '   D freezing because of the loss imbalance'+'\n'
            
        if np.isnan(Dis_train_his.history['loss']):
            curr_log = curr_log + 'Training interrupted'
            break

        ##### G training rounds
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

            # Draw D loss correspond to real and fake data
            generated_list = Generator.predict(input_noise)
            Dis_real_data = data_Dis(sine_data,None,data_params,network_params)
            Dis_fake_data = data_Dis(None,generated_list,data_params,network_params)
            curr_real_loss = Discriminator.evaluate_generator(generator=Dis_real_data)
            curr_fake_loss = Discriminator.evaluate_generator(generator=Dis_fake_data)

            Dreal_loss.append(curr_real_loss)
            Dfake_loss.append(curr_fake_loss)
            D_eval_idx.append(G_idx[-1])

        if i%20 == 0:
            plotdata = {'eval_idx': D_eval_idx, 'Dreal_loss':Dreal_loss, 'Dfake_loss':Dfake_loss}
            draw_D_loss(plotdata,path,i)

        if np.isnan(Gen_train_his.history['loss']):
            curr_log = curr_log + 'Training interrupted'
            break

		# Draw D & G loss curve after every 20 epochs
        if i%20 == 0:
            plotdata = {'G_loss':Gen_loss,'D_loss':Dis_loss,'G_idx':G_idx,'D_idx':D_idx}
            draw_train_loss(plotdata,path,i)

        # Save model weights every 100 epochs
        if i%100 == 0:
            Gen_wfile = 'GW_epoch_'+str(i)+'.hdf5'
            Dis_wfile = 'DW_epoch_'+str(i)+'.hdf5'
            Generator.save_weights(os.path.join(path,Gen_wfile), True)
            Discriminator.save_weights(os.path.join(path,Dis_wfile), True)

		# Plot generated examples every 50 epochs
        if i%50 == 0:
            generated_list = Generator.predict(input_noise)
            draw_generated_signal(generated_list,path,epoch=i)
            curr_log = curr_log + '   Generated example saved'+'\n'

		# Write log file
        with open(os.path.join(path,Training_log_file),'a') as filehandle:
            filehandle.write(curr_log)
            filehandle.write('Training time: '+ str((time.time()-start_time)/60) +' min \n')
            filehandle.write('\n')
            filehandle.close()
    
    scipy.io.savemat(os.path.join(path,'Training_history.mat'),{'G_loss':Gen_loss, 'D_loss':Dis_loss, 'G_idx': G_idx, 'D_idx': D_idx})

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

        progress_bar = Progbar(target=nb_batch)
        curr_log = 'Epoch ' +str(epoch)+': ' + '\n'
        print(colored('Training epoch '+str(epoch),'yellow'))

        if epoch%100 == 0:
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
                    
                    plt.plot(np.arange(data_params['data_len']),batch_signal[5,:])
                    plt.savefig(os.path.join(path,'dis data generation example.png'))
                    plt.close

                    batch_real_label = np.ones((batch_size,data_params['data_len'],1))
                    batch_noise = np.random.normal(0,1,size=[batch_size,data_params['data_len'],network_params['latent_dim']])

                    batch_generated = Generator.predict(batch_noise)

                    batch_input = np.concatenate((batch_signal,batch_generated))
                    batch_label = np.concatenate((batch_real_label,np.zeros((batch_size,data_params['data_len'],1))))
                    Discriminator.train_on_batch(batch_input,batch_label)
            

                for g in range(G_rounds):
                    batch_noise = np.random.normal(0,1,size=[batch_size,data_params['data_len'],network_params['latent_dim']])
                    batch_trick = np.ones((batch_size,data_params['data_len'],1))

                    Combined.train_on_batch(batch_noise,batch_trick)

                progress_bar.update(index + 1)
                    
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

        
        # Evaluate D loss in real and fake dataset respectively
        Dreal_loss.append(Discriminator.evaluate(real_data,np.ones((data_params['data_size'],data_params['data_len'],1))))
        Dfake_loss.append(Discriminator.evaluate(generated_data,np.zeros((data_params['data_size'],data_params['data_len'],1))))

        # Draw loss curves after every 20 epochs
        if epoch%20 == 0:
            plotdata = {'eval_idx': Idx, 'Dreal_loss':Dreal_loss, 'Dfake_loss':Dfake_loss}
            draw_D_loss(plotdata,path,epoch)
            plotdata_GD = {'G_loss':Gen_loss,'D_loss':Dis_loss,'G_idx':Idx,'D_idx':Idx}
            draw_train_loss(plotdata_GD,path,epoch)
            curr_log = curr_log + 'Loss curve generated'+'\n'
		
        # Save model weights every 100 epochs
        if epoch%100 == 0:
            Gen_wfile = 'GW_epoch_'+str(epoch)+'.hdf5'
            Dis_wfile = 'DW_epoch_'+str(epoch)+'.hdf5'
            Generator.save_weights(os.path.join(path,Gen_wfile), True)
            Discriminator.save_weights(os.path.join(path,Dis_wfile), True)

		# Plot generated examples every 50 epochs
        if epoch%50 == 0:
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
        optimizer=Adam(lr=D_lr),
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
    

    curr_path = 'D1G3_TOB_Adamlr0.001_decay0.707'
    #Discriminator.load_weights(os.path.join(curr_path,'DW.hdf5'))
    #Generator.load_weights(os.path.join(curr_path,'GW.hdf5'))
    #Combined.set_weights(Generator.get_weights())

    train_RGAN_batch(data_params,network_params,real_data=sine_data,G_rounds=3,D_rounds=1,path=curr_path,begin_epoch=1,end_epoch=1000,G_lr=G_lr,D_lr=D_lr,decay=0.707,Pre_loss_file=None)
