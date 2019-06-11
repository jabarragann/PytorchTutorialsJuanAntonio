import pandas as pd
import numpy as np
import time

class DataContainer:

    def __init__(self):
        self.dataWindowArray = []
        self.length = 0

    def createWindow(self,beginTime, endTime, shimmerSize, epocSize):
        self.dataWindowArray.append(DataWindow(beginTime, endTime, shimmerSize, epocSize))
        self.length += 1

    def fillData(self, shimmerData=None, epocData=None):
        beginTime = time.time()

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


        endTime = time.time()
        print(endTime - beginTime)
        print("Finish")


        # # Fill Epoc Data
        # for i in range(epocData.shape[0]):
        #     while epocData['COMPUTER_TIME'][i] > self.dataWindowArray[windowIdx].endTime:
        #         windowIdx += 1
        #
        #         if windowIdx >= self.length:
        #             break
        #
        #     if windowIdx >= self.length:
        #         break
        #     self.dataWindowArray[windowIdx].epocArray[i] = epocData[['PPG','COMPUTER_TIME','LABEL']].values[i]

        print("Finish")
    def createMetrics(self):
        self.calculateLabel()
        self.createMetrics()

    def calculateLabel(self):
        for i in range(len(self.dataWindowArray)):
            totalLength = self.dataWindowArray[i].epocArray[:,2].shape[0]
            label = sum(self.dataWindowArray[i].epocArray[:,2]) / totalLength
            self.dataWindowArray[i].globalLabel = int(label)

    def createSpectogram(self):
        pass



class DataWindow:

    def __init__(self, beginTime, endTime, shimmerSize, epocSize):

        self.beginTime = float(beginTime)
        self.endTime = float(endTime)

        self.shimmerArray = np.zeros((shimmerSize, 1+2))
        self.epocArray = np.zeros((epocSize, 14+2))
        self.actualShimmerSize = 0
        self.actualEpocSize = 0

        self.spectogramVolume = None
        self.globalLabel = None

    def createSpectogramVolume(self):
        pass

    def createGlobalLabel(self):
        pass


TRIAL = 2
if __name__ == '__main__':

    shimmerFile = pd.read_csv('./Data/S1_T{:d}_fusion_shimmer.txt'.format(TRIAL), sep=' ')
    epocFile = pd.read_csv('./Data/S1_T{:d}_fusion_epoc.txt'.format(TRIAL), sep=' ')
    fiveSecondWindowStamps = pd.read_csv("./Data/S1_T{:d}_TimestampEvery5Seconds.txt".format(TRIAL)
                                         , sep=' ').values

    container = DataContainer()

    for i in range(fiveSecondWindowStamps.shape[0] - 1):
        container.createWindow(fiveSecondWindowStamps[i],
                               fiveSecondWindowStamps[i+1],
                               800,
                               800)

    container.fillData(shimmerData=shimmerFile, epocData = epocFile)
    container.createMetrics()
