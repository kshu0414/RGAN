import numpy as np
from matplotlib import pyplot as plt
import scipy.io

nb_samples = 1000
T = 30
time_steps = np.arange(0,T,1)
freq = 2*np.pi*np.random.uniform(low=1.0,high=5.0,size=nb_samples)
amp = np.random.uniform(low=0.1,high=0.9,size=nb_samples)
phi = np.random.uniform(low=-np.pi,high=np.pi,size=[nb_samples,1])
samples = np.multiply(amp.reshape(nb_samples,1),np.sin(np.outer(freq,time_steps)+np.repeat(phi,T,axis=1)))

scipy.io.savemat('Sinus_data.mat',{'samples':samples})