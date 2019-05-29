import pickle
import matplotlib.pyplot as plt
import numpy
import copy

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

    modelNumber = 1
    statsFile = './Model{:d}/foldAccuracies.pkl'.format(modelNumber)
    valAccuracyPlots = pickle.load(open(statsFile, 'rb'))

    final  = copy.deepcopy(valAccuracyPlots)

    #Calculate filtered output
    for i in range(5):
        movAverageFilter = MovingAverage(alpha=0.85)
        for j in range(len(valAccuracyPlots[i])):
            final[i][j] = movAverageFilter.update(valAccuracyPlots[i][j])

    #Plot values
    fig, axes = plt.subplots(2,1,sharex=True)
    for i in range(5):
        axes[0].plot(valAccuracyPlots[i], label='fold{:d}'.format(i))
        axes[1].plot(final[i], label='fold{:d}'.format(i))

    plt.set_title ="Accuracy Vs epoch"
    axes[1].set_title ="Accuracy Vs epoch filtered"
    axes[0].set_ylabel= "Accuracy"
    axes[1].set_ylabel= "Accuracy"
    axes[1].set_xlabel= "epoch"
    axes[0].grid()
    axes[1].grid()
    axes[0].legend()
    axes[1].legend()

    plt.show()