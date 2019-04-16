import torch
from torch import nn
import MyMNIST as mnist
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import pickle
import time

class ConvNetwork(nn.Module):

    def __init__(self,out_1=2,out_2=1):

        super(ConvNetwork,self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=out_1,kernel_size=5,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(out_2*4*4,10)

    def forward(self,x):

        out = self.cnn1(x)
        out = torch.nn.functional.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = torch.nn.functional.relu(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0),-1)
        out = self.fc1(out)

        return out

def showImage(x,y):
    x = x * 255
    plt.imshow(x[0].numpy().astype(dtype='uint8'), cmap='gray', vmin=0, vmax=255)
    plt.title(str(y.numpy()))
    plt.show()

if __name__ == '__main__':
    #0 Create transformations
    IMAGE_SIZE = 16
    composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

    #1 Model and dataset
    print("Loading model and training dataset")
    model = ConvNetwork(out_1=32,out_2=16)
    mnistTrain = dsets.MNIST(root='./../data', train=True, download=True, transform=composed)

    # 2 Sampler and Data loader
    split = int(0.8 * len(mnistTrain))
    idxArr = np.arange(len(mnistTrain))
    trainIdx, valIdx = idxArr[:split], idxArr[split:]
    trainSampler = SubsetRandomSampler(trainIdx)
    valSampler = SubsetRandomSampler(valIdx)

    trainLoader = DataLoader(mnistTrain, batch_size=256, sampler=trainSampler)
    validLoader = DataLoader(mnistTrain, batch_size=256, sampler=valSampler)

    # 3 Optimizer
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=1e-6, momentum=0.9, nesterov=True)

    trainLogger = mnist.TrainLogger("MNIST classifiers")
    modelName = 'cnn16-32-2'
    trainLogger.addModelInfo(modelName)
    valAccuracyHistory = []
    trLossHistory = []

    print("training model ...")

    # 4 Train
    start, end = 0,0
    for epoch in range(5):
        start = time.time()
        for x, y in trainLoader:
            optimizer.zero_grad()
            yHat = model(x)
            loss = lossFunction(yHat, y)
            loss.backward()
            optimizer.step()

            trLossHistory.append(loss.item())

        correct = 0
        total = 0
        end = time.time()
        print("Time per epoc: {:.5f}".format(end-start))
        for x, y in validLoader:
            z = model(x)
            yHat = z.argmax(1)
            correct += (yHat == y).sum().item()
            total += y.shape[0]

        accuracy = correct / total
        valAccuracyHistory.append(accuracy)
        print("Accuracy in validation in epoch {:d} set: {:.6f}".format(epoch, accuracy))

    print("Accuracy in validation in epoch {:d} set: {:.6f}".format(epoch, accuracy))
    torch.save(model.state_dict(), './_MNISTModels/' + modelName + '.pt')

    # Add Train information to data logger
    trainLogger.dict[modelName]['trainLoss'] = trLossHistory
    trainLogger.dict[modelName]['validationAccuracy'] = valAccuracyHistory

    with open('./_MNISTModels/' + modelName + '_log.pt', 'wb') as outFile:
        pickle.dump(trainLogger.dict, outFile)

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(trLossHistory)
    axes[1].plot(valAccuracyHistory)
    plt.show()
    print("Finish")