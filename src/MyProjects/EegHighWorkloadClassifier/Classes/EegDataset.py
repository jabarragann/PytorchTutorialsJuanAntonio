from torch.utils.data import Dataset, DataLoader
import torch
from os.path import isfile, join
from os import listdir
import pickle


class EegDataset(Dataset):

    def __init__(self,datasetPath="./Dataset/",transform =None):

        self.datasetPath = datasetPath
        self.files = [join(datasetPath, f) for f in listdir(datasetPath) if isfile(join(datasetPath, f))]
        self.len = len(self.files)

        self.x = torch.zeros((self.len, 14, 51, 7))
        self.y = torch.zeros(len(self.files),dtype=torch.long)

        self.positiveLength = 0
        self.negativeLength = 0
        self.positiveIdx = []
        self.negativeIdx = []

        for i in range(len(self.files)):

            with open(self.files[i],'rb') as f1:
                dataDict = pickle.load(f1)
                self.x[i, :, :, :] = torch.torch.from_numpy(dataDict['data'])
                self.y[i] = int(dataDict['label'])

                if self.y[i] == 1:
                    self.positiveLength += 1
                    self.positiveIdx.append(i)
                elif self.y[i] == 0:
                    self.negativeLength += 1
                    self.negativeIdx.append(i)


        # Normalizin Values
        self.xMean = self.x.mean()
        self.xStd  = self.x.std()
        self.arbitraryScaling = self.x.max()*3/8
        self.xMax = self.x.max()
        self.xMin = self.x.min()

        # Scaling by (max - min)
        self.x = self.x / (self.xMax - self.xMin + 1e-7)

        #Scaling by mean and std
        #self.x = (self.x - self.xMean) / (self.xStd + 1e-7)

        #Scaling with arbitrary scaling factor
        # self.x = self.x / (self.arbitraryScaling)


        self.transform = transform


    def __getitem__(self, index):

        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.len
