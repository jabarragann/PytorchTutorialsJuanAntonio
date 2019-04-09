

import torch
import MyMNIST as mnist
import LogisticRegression as lr
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

if __name__ == '__main__':

    mnistTest  = mnist.MyMNIST(train=False)
    model = lr.LogisticRegression(28*28,10)
    model.load_state_dict(torch.load('./_MNISTModels/bestModel3.pt'))
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








