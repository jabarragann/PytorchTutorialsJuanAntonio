import os
import numpy as np
import struct
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

class LogisticRegression(nn.Module):

    def __init__(self, in_features, out):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(in_features, out)

    def forward(self, x):
        x = self.linear(x)
        out = torch.sigmoid(x)
        return out

class MyMINST(Dataset):

    def __init__(self, train = True):
        if train:
            self.x, self.y = self.loadMINST(dataset="training", path='./data/raw')
            self.x, self.y = torch.from_numpy(self.x), torch.from_numpy(self.y)
        else:
            self.x, self.y = self.loadMINST(dataset="testing", path='./data/raw')
            self.x, self.y = torch.from_numpy(self.x), torch.from_numpy (self.y)

        self.x = self.x.reshape(-1,28*28)
        self.len = self.x.shape[0]

        #Normalize Data
        # self.mean = self.x.mean(dim=0).reshape(-1,784)
        # self.std = self.x.std(dim=0).reshape(-1,784)

        self.x = (self.x - 127.5) / 255

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

    def loadMINST(self, dataset='training', path='.'):

        if dataset == "training":
            imagesPath = os.path.join(path, 'train-images-idx3-ubyte')
            labelsPath = os.path.join(path, 'train-labels-idx1-ubyte')
        elif dataset == "testing":
            imagesPath = os.path.join(path, 't10k-images-idx3-ubyte')
            labelsPath = os.path.join(path, 't10k-labels-idx1-ubyte')

        # #Read images
        with open(imagesPath,'rb') as imageFile:
            magic = struct.unpack('>I', imageFile.read(4))[0]
            size = struct.unpack('>I',imageFile.read(4))[0]
            rows = struct.unpack('>I',imageFile.read(4))[0]
            cols = struct.unpack('>I', imageFile.read(4))[0]

            totalBytes = size*rows*cols
            imageArray = 255 - \
                         np.asarray(struct.unpack('>' + 'B' * totalBytes, \
                         imageFile.read(totalBytes)),dtype='float32').reshape((size, rows, cols))

        #Read labels
        with open(labelsPath, 'rb') as labelsFile:
            magicLabel = struct.unpack('>I', labelsFile.read(4))[0]
            sizeLabel = struct.unpack('>I', labelsFile.read(4))[0]
            labelArray = np.asarray(struct.unpack('>' + 'B' * sizeLabel, \
                         labelsFile.read(sizeLabel)),dtype='int64')

            #oneHotEncodedLabels = np.eye(10,dtype='float32')[labelArray]

        return imageArray, labelArray

    def showImage(self, idx):
        temp = self.x[idx]
        temp = temp * 255 +127.5
        temp = temp.reshape((28,28))
        plt.imshow(temp.numpy().astype(dtype='uint8'), cmap='gray', vmin=0, vmax=255)
        #one hot encoding
        # label = self.y[idx].argmax()
        # plt.title(str(label.numpy()))
        #normal
        plt.title(str(self.y.numpy()))
        plt.show()
        return


print("Loading Data...")

#Load Data
mnistTrain = MyMINST(train=True)
#mnistTest  = MyMINST(train=False)

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
model = LogisticRegression(inputDim,outputDim)

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

trainLossHistory = []
accuracyHistory = []

print("Training...")
for epoch in range(15):
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


print("Accuracy in validation set:",accuracy)
torch.save(model.state_dict(),'./_model/bestModel.pt')
fig, axes = plt.subplots(2,1)
axes[0].plot(trainLossHistory)
axes[1].plot(accuracyHistory)
plt.show()
print("Finish")

#Pending at the cross validation procedure to set parameters like the learning rate.
#Create the file the test the final model with the test set.
#Test running this same file in a computer with CUDA. (ISAT Deep)