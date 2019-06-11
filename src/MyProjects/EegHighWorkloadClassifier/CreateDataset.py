import pandas as pd
import pickle
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import cv2



if __name__ == '__main__':
    trial = 7
    dataset = 4
    epocFile = open('./Data/S1_T{:d}_fusion_epoc.txt'.format(trial), 'r')
    shimmerFile = open('./Data/S1_T{:d}_fusion_shimmer.txt'.format(trial), 'r')

    epocData = pd.read_csv(epocFile, sep=' ').values
    shimmerData = pd.read_csv(shimmerFile, sep=' ').values

    initIdx = 0
    finalIdx = 0
    counter = 0
    posCount = 0
    negCount = 0

    for i in range(epocData.shape[0]):
        if epocData[i, -2] > 0:
            initIdx = i

        if (i+1) != epocData.shape[0]:
            if epocData[i + 1, -2] > 0:
                finalIdx = i

                spectogramVolume = np.zeros((14,51,7))
                for i in range(14):
                    x = epocData[initIdx:finalIdx + 1, i]

                    f, t, Sxx = signal.spectrogram(x, 1, nperseg=100, mode = 'magnitude')
                    #print(Sxx.max())
                    Sxx = cv2.resize(Sxx, dsize=(7, 51))
                    spectogramVolume[i, :, :] = Sxx

                print(str(finalIdx - initIdx + 1))
                avgLabel = np.average(epocData[initIdx:finalIdx + 1, -1])
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

