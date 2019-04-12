import torch
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

#If the code is ran from pycharm
import MyMNIST as mnist
#Else if the code is ran from somewhere else
# import _MNISTClasses.MyMNIST as MyMNIST
# import _MNISTClasses.LogisticRegression as LogisticRegression
from torch import nn
import NeuralNetwork as mynn


class NeuralNetwork (nn.Module):
    def __init__(self, input,hidden,output):
        super().__init__()
        self.hiddenLayer = nn.Linear(input,hidden)
        self.outputLayer = nn.Linear(hidden,output)

    def forward(self, x):
        x = self.hiddenLayer(x)
        x = torch.sigmoid(x)
        x = self.outputLayer(x)
        return x

if __name__ == '__main__':

    modelName = 'sigmoid_nn'
    with open('./_MNISTModels/'+modelName+'_log.pt', 'rb') as infile:
        dict = pickle.load(infile)

    mnistTest  = mnist.MyMNIST(train=False)
    model = mynn.NeuralNetwork(dict[modelName]['layers'])
    model.load_state_dict(torch.load('./_MNISTModels/'+modelName+'.pt'))
    model.eval()

    randomSampler = SubsetRandomSampler(range(len(mnistTest)))
    dataBatches = DataLoader(mnistTest,batch_size=1000,sampler=randomSampler)

    accuracyArr = []
    totalCorrect = 0
    for x,y in dataBatches:

        yHat = model(x)
        yHat = yHat.argmax(1)
        correct = (y == yHat).sum().item()
        accuracyArr.append(correct/y.shape[0])

        totalCorrect += correct

    print("Accuracy: {:0.4f}".format(totalCorrect/len(mnistTest)))

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(dict[modelName]['trainLoss'])
    axes[1].plot(dict[modelName]['validationAccuracy'])
    plt.show()
    print("Finish")








