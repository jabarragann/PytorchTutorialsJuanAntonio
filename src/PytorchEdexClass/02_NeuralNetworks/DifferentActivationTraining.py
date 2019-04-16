import MyMNIST as mnist
import torch
from torch import nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import pickle
import NeuralNetwork as mynn
import MyMNIST as mnist

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

    #1 Model and Dataset
    print("loading data ...")
    mnistTrain = mnist.MyMNIST(train=True)
    layers = [28*28,150,10]
    activationArr = ['sigmoid','tanh','relu']

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

    trainLogger = mnist.TrainLogger("MNIST classifiers")

    for idx, actFunct in enumerate(activationArr):
        model = mynn.NeuralNetwork(layers, actFunct)
        optimizer = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=1e-6, momentum=0.9, nesterov=True)
        modelName = actFunct+'_nn'
        trainLogger.addModelInfo(modelName)
        valAccuracyHistory =[]
        trLossHistory = []

        print("training {} model ...".format(modelName))
        #4 Train
        for epoch in range(50):
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
        torch.save(model.state_dict(), './_DiffActivationComp/'+modelName+'.pt')

        #Add Train information to data logger
        trainLogger.dict[modelName]['layers'] = layers
        trainLogger.dict[modelName]['activation'] = actFunct
        trainLogger.dict[modelName]['trainLoss'] =  trLossHistory
        trainLogger.dict[modelName]['validationAccuracy'] =  valAccuracyHistory

    with open('./_DiffActivationComp/'+'trainLogger'+'_log.pt','wb') as outFile:
        pickle.dump(trainLogger.dict, outFile)


    fig, axes = plt.subplots(2, 1)
    for modelName in trainLogger.dict.keys():
        trLossHistory = trainLogger.dict[modelName]['trainLoss']
        valAccuracyHistory = trainLogger.dict[modelName]['validationAccuracy']
        axes[0].plot(trLossHistory,label = modelName)
        axes[1].plot(valAccuracyHistory,label = modelName)

    axes[0].legend()
    axes[1].legend()
    plt.show()
    print("Finish")





