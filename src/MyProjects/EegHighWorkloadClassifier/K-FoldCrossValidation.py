import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from os import listdir
from os.path import isfile, join
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import EegDataset
import EegConvNetwork


def save_checkpoint(optimizer, model, epoch, trLossH, valLossH, valAccH, filename):
    checkpoint_dict = {'optimizer': optimizer.state_dict(),
                       'model': model.state_dict(),
                       'epoch': epoch,
                       'trLossH': trLossH,
                       'valLossH': valLossH,
                       'valAccH': valAccH}

    torch.save(checkpoint_dict, filename)


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename, map_location=DEVICE)
    epoch = checkpoint_dict['epoch']
    trLossH = checkpoint_dict['trLossH']
    valLossH = checkpoint_dict['valLossH']
    valAccH = checkpoint_dict['valAccH']

    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    return epoch, trLossH, valLossH, valAccH


def fiveKFoldTraining(trainData, folds, cnn, totalEpochs, modelNumb):

    valAccuracyPlots = []
    loss = 0
    for i in range(5):
        #Create Data loaders
        trainLoader = DataLoader(trainData, batch_size=16, sampler=folds[i]['train'])
        validLoader = DataLoader(trainData, batch_size=16, sampler=folds[i]['val'])

        #Create the model, optimizer and loss function
        torch.manual_seed(RANDOM_SEED)
        model = EegConvNetwork.ConvNetwork(out_1=cnn['CONV_LAYER1'],
                                           out_2=cnn['CONV_LAYER2'],
                                           out_3=cnn['CONV_LAYER3'],
                                           activation=cnn['ACTIVATION'])

        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

        #TrainLogger
        valAccuracyHistory = []
        valLossHistory = []
        trLossHistory = []

        #Train Fold
        #Reset Max accuracy
        maxValidationAccuracy = 0.7

        for epoch in range(totalEpochs):
            for x, y in trainLoader:
                optimizer.zero_grad()
                yHat = model(x)
                loss = lossFunction(yHat, y)
                loss.backward()
                optimizer.step()
                trLossHistory.append(loss.item())

            correct = 0
            total = 0
            for x, y in validLoader:
                z = model(x)
                yHat = z.argmax(1)
                correct += (yHat == y).sum().item()
                total += y.shape[0]

            accuracy = correct / total
            valAccuracyHistory.append(accuracy)

            if epoch%25 == 0 and epoch != 0:
                #print("Loss in Training in epoch {:d} set:       {:.5f}".format(epoch, loss))
                print("Accuracy in validation in epoch {:d} set: {:.5f}".format(epoch, accuracy))

            # Save Check Point
            if accuracy > maxValidationAccuracy:
                maxValidationAccuracy = accuracy
                checkpoint_filename = MODELS_DIRECTORY + 'Model{:d}/classifier-f{:02d}-e{:03d}.pkl'.format(modelNumb, i, epoch)
                save_checkpoint(optimizer, model, epoch, trLossHistory, valLossHistory, valAccuracyHistory,
                                checkpoint_filename)

                checkpoint_txt = MODELS_DIRECTORY + 'Model{:d}/classifier-f{:02d}-e{:03d}-acc{:0.5f}.txt'.format(modelNumb, i, epoch, maxValidationAccuracy)
                with open(checkpoint_txt,'w') as f:
                    pass

                maxValidationAccuracy = accuracy

        maxValAccPerFold =  max(valAccuracyHistory)
        valAccuracyPlots.append(valAccuracyHistory)
        print("Max Validation Accuracy fold {:02d}:        {:0.5f}\n".format(i, maxValAccPerFold))


    #Save data Accuracies
    statsFile = MODELS_DIRECTORY + 'Model{:d}/foldAccuracies.pkl'.format(modelNumb)
    with open(statsFile, 'wb') as f:
        pickle.dump(valAccuracyPlots, f)

    #Print training summary
    for i in range(5):
        print("Max accuracy in fold {:d}: {:.4f}".format(i,max(valAccuracyPlots[i])))

    averageMaxAcc = sum([max(valAccuracyPlots[i]) for i in range(5)])/5
    print("Average Max Accuracy: {:.5f}".format(averageMaxAcc))


def createModelDirectory(modelNumber):
    path = MODELS_DIRECTORY + "Model{:02d}".format(modelNumber)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


#Cnn Architectures -- Best Model cnn8!!
cnn1 = {"CONV_LAYER1": 32, "CONV_LAYER2": 64, "CONV_LAYER3": 12, "ACTIVATION": 'selu'}
cnn2 = {"CONV_LAYER1": 32, "CONV_LAYER2": 64, "CONV_LAYER3": 12, "ACTIVATION": 'elu'}
cnn3 = {"CONV_LAYER1": 32, "CONV_LAYER2": 64, "CONV_LAYER3": 12, "ACTIVATION": 'relu'}
cnn4 = {"CONV_LAYER1": 16, "CONV_LAYER2": 32, "CONV_LAYER3": 6, "ACTIVATION": 'selu'}
cnn5 = {"CONV_LAYER1": 16, "CONV_LAYER2": 32, "CONV_LAYER3": 6, "ACTIVATION": 'elu'}
cnn6 = {"CONV_LAYER1": 16, "CONV_LAYER2": 32, "CONV_LAYER3": 6, "ACTIVATION": 'relu'}
cnn7 = {"CONV_LAYER1": 64, "CONV_LAYER2": 128, "CONV_LAYER3": 24, "ACTIVATION": 'selu'}
cnn8 = {"CONV_LAYER1": 64, "CONV_LAYER2": 128, "CONV_LAYER3": 24, "ACTIVATION": 'elu'}
cnn9 = {"CONV_LAYER1": 64, "CONV_LAYER2": 128, "CONV_LAYER3": 24, "ACTIVATION": 'relu'}


#Global variables
DEVICE = 'cpu'
MODELS_DIRECTORY = './Models/'

#RandomSeed
#Model 2 and Model 3
#RANDOM_SEED = 742
#Model 4
RANDOM_SEED = 480


if __name__ == '__main__':

    #Create Training data object
    trainData = EegDataset.EegDataset(datasetPath="./Dataset/D4")
    print("Number of negative examples ",trainData.negativeLength)
    print("Number of positive examples ", trainData.positiveLength)

    #Shuffle training data
    split = int(0.2 * len(trainData))
    np.random.seed(seed=RANDOM_SEED)
    idxArr = np.random.permutation(len(trainData))

    #Create 5 folds for cross Validation
    folds = []
    total = len(trainData)

    for i in range(5):
        valRange = range(split*i, split*(i+1) if split*(i+1) < total else total)
        valIdx = idxArr[valRange]
        trainIdx = np.delete(idxArr, valRange)
        trainSampler = SubsetRandomSampler(trainIdx)
        valSampler = SubsetRandomSampler(valIdx)
        folds.append({'train': trainSampler, 'val': valSampler})

    #Train Models
    totalEpochs = 50
    print("Start training")
    #cnnArch = [cnn1,cnn2,cnn3,cnn4,cnn5,cnn6,cnn7,cnn8,cnn9]
    cnnArch = [cnn8]
    modelNumber = 30

    for cnn in cnnArch:
        createModelDirectory(modelNumber)
        fiveKFoldTraining(trainData, folds, cnn, totalEpochs, modelNumber)
        modelNumber += 1



    # fig, ax  = plt.subplots(1,1)
    # for i in range(5):
    #     ax.plot(valAccuracyPlots[i], label ="fold {:d}".format(i) )
    #     print("Max accuracy in fold {:d}: {:.4f}".format(i,max(valAccuracyPlots[i])))
    #
    # averageMaxAcc = sum([max(valAccuracyPlots[i]) for i in range(5)])/5
    # print("Average Max Accuracy: {:.5f}".format(averageMaxAcc))
    # ax.set_title = "Accuracy vs Epoch for every fold."
    # ax.set_ylabel("Accuracy")
    # ax.set_xlabel("Epoch")
    # ax.legend()
    #
    # plt.show()


    # print(trainData.positiveLength, trainData.negativeLength)
    # np.random.seed(seed=742)
    # print(np.random.permutation([11, 24, 49, 12, 15, 58, 49, 12, 55]))
    # np.random.seed(seed=742)
    # print(np.random.permutation([11, 24, 39, 12, 15, 58, 49, 12, 55]))
    #
    # testArr = [1, 4, 1, 2, 1, 5, 1]
    # remove = [0, 2, 4, 6]
    # newTest = np.delete(testArr, remove)
    # print(newTest)
    # print(testArr)