import MyMNIST as mnist
import torch
from torch import nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import pickle

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

class TrainLogger:
    def __init__(self,name):
        self.name = name
        self.dict = {}

    def addModelInfo(self,name):
        self.dict[name] = {'trainLoss':[],'validationLoss':[],'validationAccuracy':[]}


if __name__ == '__main__':

    #1 Model and Dataset
    print("loading data ...")
    mnistTrain = mnist.MyMNIST(train=True)
    model = NeuralNetwork(28*28,100, 10)

    #2 Sampler and Data loader
    split = int(0.8 * len(mnistTrain))
    idxArr = np.arange(len(mnistTrain))
    trainIdx, valIdx = idxArr[:split], idxArr[split:]
    trainSampler = SubsetRandomSampler(trainIdx)
    valSampler = SubsetRandomSampler(valIdx)

    trainLoader = DataLoader(mnistTrain,batch_size=256,sampler=trainSampler)
    validLoader = DataLoader(mnistTrain, batch_size=256, sampler=valSampler)

    #3 Optimizer
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(),lr=0.1, weight_decay= 1e-6, momentum = 0.9, nesterov = True)

    trainLogger = TrainLogger("MNIST classifiers")
    modelName = 'sigmoid_nn'
    trainLogger.addModelInfo(modelName)
    valAccuracyHistory =[]
    trLossHistory = []

    print("training model ...")
    #4 Train
    for epoch in range(45):
        for x,y in trainLoader:
            optimizer.zero_grad()
            yHat = model(x)
            loss = lossFunction(yHat,y)
            loss.backward()
            optimizer.step()

            trLossHistory.append(loss.item())

        correct = 0
        total = 0
        for x,y in validLoader:
            z = model(x)
            yHat = z.argmax(1)
            correct += (yHat==y).sum().item()
            total += y.shape[0]

        accuracy = correct / total
        valAccuracyHistory.append(accuracy)
        print("Accuracy in validation in epoch {:d} set: {:.6f}".format(epoch,accuracy))

    print("Accuracy in validation in epoch {:d} set: {:.6f}".format(epoch, accuracy))
    torch.save(model.state_dict(), './_MNISTModels/'+modelName+'.pt')

    #Add Train information to data logger
    trainLogger.dict[modelName]['trainLossArray'] =  trLossHistory
    trainLogger.dict[modelName]['validationAccuracy'] =  valAccuracyHistory
    with open('./_MNISTModels/'+modelName+'_log.pt','wb') as outFile:
        pickle.dump(trainLogger, outFile)


    fig, axes = plt.subplots(2, 1)
    axes[0].plot(trLossHistory)
    axes[1].plot(valAccuracyHistory)
    plt.show()
    print("Finish")





