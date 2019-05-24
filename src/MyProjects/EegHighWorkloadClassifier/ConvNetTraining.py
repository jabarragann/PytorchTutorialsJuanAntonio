import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import EegDataset
import EegConvNetwork


CONV_LAYER1 = 32
CONV_LAYER2 = 64
CONV_LAYER3 = 12
IMAGE_SIZE = 4

if __name__ == '__main__':

    trainData = EegDataset.EegDataset()

    model = EegConvNetwork.ConvNetwork(out_1=CONV_LAYER1, out_2=CONV_LAYER2, out_3=CONV_LAYER3)

    # 2 Sampler and Data loader
    split = int(0.8 * len(trainData))
    idxArr = np.random.permutation(len(trainData))
    trainIdx, valIdx = idxArr[:split], idxArr[split:]
    trainSampler = SubsetRandomSampler(trainIdx)
    valSampler = SubsetRandomSampler(valIdx)

    trainLoader = DataLoader(trainData, batch_size=16, sampler=trainSampler)
    validLoader = DataLoader(trainData, batch_size=16, sampler=valSampler)

    # #TEST CONV NET FORWARD PASS
    # trainLoader = iter(trainLoader)
    # x, y = next(trainLoader)
    # model.forward(x)

    # 3 Optimizer
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

    trainLogger = {}
    modelName = 'cnn_batch_normalization'
    trainLogger[modelName] = {'CONV_LAYER1': CONV_LAYER1, 'CONV_LAYER2': CONV_LAYER2, 'IMAGE_SIZE': IMAGE_SIZE}
    valAccuracyHistory = []
    trLossHistory = []

    print("training model ...")
    # 4 Train
    start, end = 0, 0
    for epoch in range(150):
        start = time.time()
        for x, y in trainLoader:
            optimizer.zero_grad()
            yHat = model(x)
            loss = lossFunction(yHat, y)
            loss.backward()
            optimizer.step()

            trLossHistory.append(loss.item())

        print("Loss in Training in epoch {:d} set: {:.6f}".format(epoch, loss))
        correct = 0
        total = 0
        end = time.time()
        print("Time per epoc: {:.5f}".format(end - start))
        for x, y in validLoader:
            z = model(x)
            yHat = z.argmax(1)
            correct += (yHat == y).sum().item()
            total += y.shape[0]

        accuracy = correct / total
        valAccuracyHistory.append(accuracy)
        print("Accuracy in validation in epoch {:d} set: {:.6f}".format(epoch, accuracy))


    print("Accuracy in validation in epoch {:d} set: {:.6f}".format(epoch, accuracy))
    torch.save(model.state_dict(), './Model1/' + modelName + '.pt')

    # Add Train information to data logger
    trainLogger[modelName]['trainLoss'] = trLossHistory
    trainLogger[modelName]['validationAccuracy'] = valAccuracyHistory

    with open('./Model1/' + modelName + '_log.pt', 'wb') as outFile:
        pickle.dump(trainLogger, outFile)

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(trLossHistory)
    axes[1].plot(valAccuracyHistory)
    plt.show()
    print("Finish")

