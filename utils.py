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
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples)
    return samples
        
def draw_train_loss(plotdata,plot_name):
    D_line, = plt.plot(plotdata['D_idx'],np.log10(np.array(plotdata["D_loss"])),'b',label='Discriminator loss') 
    G_line, = plt.plot(plotdata['G_idx'],np.log10(np.array(plotdata["G_loss"])),'r',label='Generator loss') 
    plt.legend(handles=[D_line,G_line])
    
    plt.xlabel('Epoch')
    plt.ylabel('log10(Loss)')
    plt.savefig(plot_name)
    plt.close()

def draw_generated_signal(plotdata,path,epoch):
    data1 = plotdata[400,:]
    data2 = plotdata[800,:]
    data3 = plotdata[1200,:]

    plt.subplot(311)
    plt.plot(np.arange(len(data1)),data1)
    plt.subplot(312)
    plt.plot(np.arange(len(data2)),data2)
    plt.subplot(313)
    plt.plot(np.arange(len(data3)),data3)
    plt.suptitle('generated samples after '+str(epoch+1)+' epoch')

    plotname = 'Generated_signal_'+str(epoch+1)+'_epoch.png'
    plt.savefig(os.path.join(path,plotname))
    plt.close()
