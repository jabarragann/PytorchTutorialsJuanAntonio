import os
import numpy as np
import struct
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch

class MyMINST(Dataset):

    def __init__(self, train = True):
        if train:
            self.x, self.y = self.loadMINST(dataset="training", path='./data/raw')
            self.x, self.y = torch.from_numpy(self.x), torch.from_numpy(self.y)
        else:
            self.x, self.y = self.loadMINST(dataset="testing", path='./data/raw')
            self.x, self.y = torch.from_numpy(self.x), torch.from_numpy (self.y)

        self.len = self.x.shape[0]

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
                         labelsFile.read(sizeLabel)),dtype='float32').reshape((sizeLabel,1))

        return imageArray, labelArray

    def showImage(self, idx):
        plt.imshow(self.x[idx].numpy, cmap='gray', vmin=0, vmax=255)
        plt.title(str(self.y[idx].numpy()[0]))
        plt.show()
        return


print("Loading Data...")
#minstTrain = MyMINST(train=True)
minstTest  = MyMINST(train=False)


print("Finish")

