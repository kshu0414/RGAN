import numpy as np
import matplotlib.pyplot as plt 
import keras.backend as K
import keras.layers
import os

def sine_wave(seq_length=30, num_samples=28*5*100, num_signals=1,
        freq_low=1, freq_high=5, amplitude_low = 0.1, amplitude_high=0.9, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            # f = np.random.uniform(low=freq_low, high=freq_high)     # frequency
            f = np.random.randint(low=freq_high, high=freq_high)
            A = np.random.uniform(low=amplitude_low, high=amplitude_high)        # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset)) # sampling frequency is 30Hz
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples)
    return samples
        
def draw_train_loss(plotdata,path,epoch):
    D_line, = plt.plot(plotdata['D_idx'],np.array(plotdata["D_loss"]),'b',label='Discriminator loss') 
    G_line, = plt.plot(plotdata['G_idx'],np.array(plotdata["G_loss"]),'r',label='Generator loss') 
    plt.legend(handles=[D_line,G_line])
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plot_name = 'training_loss.png'

    plt.savefig(os.path.join(path,plot_name))
    plt.close()

def draw_D_loss(plotdata,path,epoch):
    real_line, = plt.plot(plotdata['eval_idx'],plotdata['Dreal_loss'],'g',label='D loss in real data')
    fake_line, = plt.plot(plotdata['eval_idx'],plotdata['Dfake_loss'],'m',label='D loss in fake data')
    plt.legend(handles=[real_line,fake_line])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plot_name = 'discriminator_loss.png'

    plt.savefig(os.path.join(path,plot_name))
    plt.close()

def draw_generated_signal(plotdata,path,epoch):
    data1 = plotdata[300,:]
    data2 = plotdata[600,:]
    data3 = plotdata[900,:]
    data4 = plotdata[1200,:]
    data5 = plotdata[1400,:]

    plt.subplot(511)
    plt.plot(np.arange(len(data1)),data1)
    plt.subplot(512)
    plt.plot(np.arange(len(data2)),data2)
    plt.subplot(513)
    plt.plot(np.arange(len(data3)),data3)
    plt.subplot(514)
    plt.plot(np.arange(len(data4)),data4)
    plt.subplot(515)
    plt.plot(np.arange(len(data5)),data5)
    plt.suptitle('generated samples after '+str(epoch)+' epoch')

    plotname = 'Generated_signal_'+str(epoch)+'_epoch.png'
    plt.savefig(os.path.join(path,plotname))
    plt.close()
