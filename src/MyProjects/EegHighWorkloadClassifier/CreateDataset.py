import pandas as pd
import pickle
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    trial = 7
    file = open('./Data/S1_T{:d}_fusion.txt'.format(trial),'r')

    data = pd.read_csv(file, sep=' ')

    data = data.values

    initIdx = 0
    finalIdx = 0
    counter = 0
    posCount = 0
    negCount = 0
    for i in range(data.shape[0]):
        if data[i,-2] > 0:
            initIdx = i

        if (i+1) != data.shape[0]:
            if data[i+1, -2] > 0:
                finalIdx = i

                spectogramVolume = np.zeros((14,51,7))
                for i in range(14):
                    x = data[initIdx:finalIdx+1,i]

                    f, t, Sxx = signal.spectrogram(x, 1, nperseg=100, mode = 'magnitude')
                    #print(Sxx.max())
                    Sxx = cv2.resize(Sxx, dsize=(7, 51))
                    spectogramVolume[i, :, :] = Sxx

                print(str(finalIdx - initIdx + 1))
                avgLabel = np.average(data[initIdx:finalIdx + 1, -1])
                avgLabel = 1.0 if avgLabel > 0.5 else 0.0

                if avgLabel is 1.0:
                    posCount += 1
                else:
                    negCount+=1

                finalDict = {'data': spectogramVolume, 'label': avgLabel}

                with open('./Dataset/S1_T{:d}_{:03d}.pickle'.format(trial,counter), 'wb') as handle:
                    pickle.dump(finalDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                counter += 1


    print(posCount, negCount)

