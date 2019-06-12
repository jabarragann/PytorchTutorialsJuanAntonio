import pandas as pd
import numpy as np
import time
import pickle
from scipy import signal
import cv2
import os

class DataContainer:

    def __init__(self):
        self.dataWindowArray = []
        self.length = 0

        # Normalizing values of whole trial
        self.epocMean = -1
        self.epocStd = -1
        self.shimmerMean = -1
        self.shimmerStd = -1

    def createWindow(self,beginTime, endTime, shimmerSize, epocSize, shimmerScaling=None,
                                                                     epocScaling=None):
        self.dataWindowArray.append(DataWindow(beginTime, endTime, shimmerSize, epocSize,shimmerScaling, epocScaling))
        self.length += 1

    def fillData(self, shimmerData=None, epocData=None):
        #Fill Shimmer Data
        windowIdx = 0
        shimmerIdx = 0
        shimmerData = shimmerData[['PPG', 'COMPUTER_TIME', 'LABEL']].values
        for i in range(shimmerData.shape[0]):
            while shimmerData[i,-2] > self.dataWindowArray[windowIdx].endTime:
                windowIdx += 1
                shimmerIdx = 0

                if windowIdx >= self.length:
                    break

            if windowIdx >= self.length:
                break

            self.dataWindowArray[windowIdx].shimmerArray[shimmerIdx] = shimmerData[i,:]
            self.dataWindowArray[windowIdx].actualShimmerSize += 1
            shimmerIdx += 1

        #Fill Epoc Data
        windowIdx = 0
        epocIdx = 0
        epocData = epocData[['AF3', 'F7', 'F3', 'FC5', 'T7',
                             'P7', 'O1', 'O2', 'P8', 'T8',
                             'FC6', 'F4', 'F8', 'AF4', 'COMPUTER_TIME',
                             'LABEL']].values

        for i in range(epocData.shape[0]):
            while epocData[i,-2] > self.dataWindowArray[windowIdx].endTime:
                windowIdx += 1
                epocIdx = 0

                if windowIdx >= self.length:
                    break

            if windowIdx >= self.length:
                break

            self.dataWindowArray[windowIdx].epocArray[epocIdx] = epocData[i,:]
            self.dataWindowArray[windowIdx].actualEpocSize += 1
            epocIdx += 1

    def createMetrics(self):
        for i in range(self.length):
            self.dataWindowArray[i].calculateLabel()
            self.dataWindowArray[i].createSpectogramVolume()

    def normalizeData(self):
        for i in range(self.length):
            self.dataWindowArray[i].normalizeData()

    def dumpWindowsToFiles(self):
        for i in range(self.length):
            self.dataWindowArray[i].createPickleFile(i)



class DataWindow:

    def __init__(self, beginTime, endTime, shimmerSize, epocSize,shimmerScaling=None, epocScaling=None):

        self.beginTime = float(beginTime)
        self.endTime = float(endTime)

        self.shimmerArray = np.zeros((shimmerSize, 1+2))
        self.epocArray = np.zeros((epocSize, 14+2))
        self.actualShimmerSize = 0
        self.actualEpocSize = 0

        self.spectogramVolume = np.zeros((14,51,7))
        self.globalLabel = None

        self.shimmerMean = shimmerScaling[0]
        self.shimmerStd = shimmerScaling[1]
        self.epocMean = epocScaling[0]
        self.epocStd = epocScaling[1]


    def createSpectogramVolume(self):
        if self.actualEpocSize > 400:
            for j in range(14):
                x = self.epocArray[:self.actualEpocSize, j]
                f, t, Sxx = signal.spectrogram(x, 1, nperseg=100, mode='magnitude')
                # print(Sxx.max())
                Sxx = cv2.resize(Sxx, dsize=(7, 51))
                self.spectogramVolume[j, :, :] = Sxx

    def normalizeData(self):
        self.epocArray[:,:14] = (self.epocArray[:,:14] - self.epocMean)/self.epocStd
        self.shimmerArray[:,0] = (self.shimmerArray[:,0] - self.shimmerMean)/self.shimmerStd

    def calculateLabel(self):
        totalLength = self.actualEpocSize + 1e-6
        label = sum(self.epocArray[:self.actualEpocSize, -1]) / totalLength
        self.globalLabel = round(label)

    def createPickleFile(self, windowIdx):
        global TRIAL
        if self.actualEpocSize > 400 and windowIdx>3:
            finalDict = {'data': self.spectogramVolume, 'label': self.globalLabel}
            with open('./Dataset/D4/S1_T{:d}_{:03d}.pickle'.format(TRIAL, windowIdx), 'wb') as handle:
                pickle.dump(finalDict, handle, protocol=pickle.HIGHEST_PROTOCOL)


TRIAL = 7
if __name__ == '__main__':

    path = "./Dataset/D{:d}".format(4)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    for TRIAL in [2,4,5,6,7]:
        shimmerFile = pd.read_csv('./Data/S1_T{:d}_fusion_shimmer.txt'.format(TRIAL), sep=' ')
        epocFile = pd.read_csv('./Data/S1_T{:d}_fusion_epoc.txt'.format(TRIAL), sep=' ')
        fiveSecondWindowStamps = pd.read_csv("./Data/S1_T{:d}_TimestampEvery5Seconds.txt".format(TRIAL)
                                             , sep=' ').values

        #Get Normalizing Values
        shimmerData = shimmerFile['PPG'].values
        epocData = epocFile[['AF3', 'F7', 'F3', 'FC5', 'T7',
                             'P7', 'O1', 'O2', 'P8', 'T8',
                             'FC6', 'F4', 'F8', 'AF4']].values

        epocMean, epocStd = epocData.mean(), epocData.std()
        shimmerMean, shimmerStd = shimmerData.mean(), shimmerData.mean()

        #Create and fill Data Container
        container = DataContainer()

        for i in range(fiveSecondWindowStamps.shape[0] - 1):
            container.createWindow(fiveSecondWindowStamps[i],
                                   fiveSecondWindowStamps[i+1],
                                   800, 800,
                                   shimmerScaling=(shimmerMean, shimmerStd),
                                   epocScaling=(epocMean,epocStd))

        container.fillData(shimmerData=shimmerFile, epocData = epocFile)
        container.normalizeData()
        container.createMetrics()
        container.dumpWindowsToFiles()

        print("Finish Creating Pickle Files from trial {:d}".format(TRIAL))
