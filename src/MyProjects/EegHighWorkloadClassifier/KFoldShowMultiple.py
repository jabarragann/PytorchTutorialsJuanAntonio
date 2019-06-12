import pickle
import matplotlib.pyplot as plt
import numpy
import copy
import EegDataset

class AverageBase(object):

    def __init__(self, value=0):
        self.value = float(value) if value is not None else None

    def __str__(self):
        return str(round(self.value, 4))

    def __repr__(self):
        return self.value

    def __format__(self, fmt):
        return self.value.__format__(fmt)

    def __float__(self):
        return self.value

class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """

    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha

    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value

if __name__ == '__main__':

    trainData = EegDataset.EegDataset(datasetPath='./Dataset/D4')
    print(trainData.negativeLength)
    print(trainData.positiveLength)


    for modelNumber in range(20,29):
        # dataset = 7
        # statsFile = './Models/D{:d}_Model{:d}/foldAccuracies.pkl'.format(dataset, modelNumber)
        statsFile = './Models/Model{:d}/foldAccuracies.pkl'.format(modelNumber)
        # statsFile = './Model{:d}/foldAccuracies.pkl'.format(modelNumber)
        valAccuracyPlots = pickle.load(open(statsFile, 'rb'))

        #Print maximum accuracies
        for i in range(5):
            print("Max accuracy in fold {:d}: {:.4f}".format(i,max(valAccuracyPlots[i])))

        averageMaxAcc = sum([max(valAccuracyPlots[i]) for i in range(5)])/5
        print("Average Max Accuracy: {:.5f}\n".format(averageMaxAcc))


