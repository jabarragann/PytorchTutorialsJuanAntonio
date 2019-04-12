
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler

#If the code is ran from pycharm
import MyMNIST as mnist
import LogisticRegression as lr

#Else if the code is ran from somewhere else
# import _MNISTClasses.MyMNIST as MyMNIST
# import _MNISTClasses.LogisticRegression as LogisticRegression


if __name__ == '__main__':

    #Run this module to train a linear classifier in the MNIST dataset
    #Load Data
    print("Loading Data...")
    mnistTrain = mnist.MyMNIST(train=True)


    #Split between training and validation data
    split = int(0.8 * len(mnistTrain))
    index_list = list(range(len(mnistTrain)))
    trainIdx, validIdx = index_list[:split], index_list[split:]

    tr_sampler = SubsetRandomSampler(trainIdx)
    val_sampler = SubsetRandomSampler(validIdx)

    batchSize = 256
    trainLoader = DataLoader(mnistTrain, batch_size=batchSize, sampler=tr_sampler)
    validLoader = DataLoader(mnistTrain, batch_size=batchSize, sampler=val_sampler)

    #Create model and optimizer
    inputDim = 28*28
    outputDim = 10
    model = lr.LogisticRegression(inputDim,outputDim)

    learning_rate = 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainLossHistory = []
    accuracyHistory = []

    print("Training...")
    for epoch in range(30):
        model.train()
        for x,y in trainLoader:
            optimizer.zero_grad()
            z = model(x)

            #Remember that the CrossEntropyLoss function works with out having to onehot encode the labels.
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

            trainLossHistory.append(loss.data)

        model.eval()
        correct = 0
        total = 0
        for x,y in validLoader:
            z = model(x)
            yHat = z.argmax(1)
            correct += (yHat==y).sum().item()
            total += y.shape[0]

        accuracy = correct / total
        accuracyHistory.append(accuracy)
        print("Accuracy in validation set:", accuracy)

    print("Accuracy in validation set:",accuracy)
    torch.save(model.state_dict(),'./_MNISTModels/bestModel4.pt')
    fig, axes = plt.subplots(2,1)
    axes[0].plot(trainLossHistory)
    axes[1].plot(accuracyHistory)
    plt.show()
    print("Finish")

    #Pending at the cross validation procedure to set parameters like the learning rate.
    #Create the file the test the final model with the test set.
    #Test running this same file in a computer with CUDA. (ISAT Deep)
    #Train linear classifier with BCELoss function