from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fs = 128
epoc = pd.read_csv("./Data/S1_T1_fusion.txt", sep=' ')

x = epoc.values[:,0]

f, t, Sxx = signal.spectrogram(x, fs, nperseg=150)
print(Sxx.shape)
plt.plot(x)
#plt.imshow(Sxx)
#plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()