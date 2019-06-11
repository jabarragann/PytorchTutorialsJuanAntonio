import pandas as pd

class DataContainer:

    def __init__(self):
        self.dataWindowArray = []
        self.length = 0

    def createWindow(self,beginTime, endTime):
        self.dataWindowArray.append(DataWindow(beginTime, endTime))
        self.length += 1

    def fillData(self, shimmerData=None, epocData=None):
        #Fill Shimmer Data
        windowIdx = 0
        for i in range(shimmerData.shape[0]):
            while shimmerData['COMPUTER_TIME'][i] > self.dataWindowArray[windowIdx].endTime:
                windowIdx += 1

            if windowIdx == self.length:
                break
            self.dataWindowArray[windowIdx].shimmerArray \
                .append(shimmerData[['PPG','COMPUTER_TIME','LABEL']].values[i])

            # Fill Epoc Data
            windowIdx = 0
            for i in range(epocData.shape[0]):
                while epocData['COMPUTER_TIME'][i] > self.dataWindowArray[windowIdx].endTime:
                    windowIdx += 1

                if windowIdx == self.length:
                    break
                self.dataWindowArray[windowIdx].shimmerArray \
                    .append(shimmerData[['PPG', 'COMPUTER_TIME', 'LABEL']].values[i])
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

    def __init__(self, beginTime, endTime):

        self.beginTime = float(beginTime)
        self.endTime = float(endTime)

        self.shimmerArray = []
        self.epocArray = []
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
        container.createWindow(fiveSecondWindowStamps[i], fiveSecondWindowStamps[i+1])

    container.fillData(shimmerData=shimmerFile, epocData = epocFile)
    container.createMetrics()
