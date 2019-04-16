import torch
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torchvision.datasets as dsets

#If the code is ran from pycharm
import MyMNIST as mnist
#Else if the code is ran from somewhere else
# import _MNISTClasses.MyMNIST as MyMNIST
# import _MNISTClasses.LogisticRegression as LogisticRegression
from torch import nn
import NeuralNetwork as mynn
import torchvision.transforms as transforms
import ConvNetwork as cnn


if __name__ == '__main__':
    # 0 Create transformations
    IMAGE_SIZE = 16
    composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

    # 1 Model and dataset
    print("Loading model and training dataset")
    model = cnn.ConvNetwork(out_1=32, out_2=16)
    mnistTest = dsets.MNIST(root='./../data', train=True, download=True, transform=composed)

    modelName = 'cnn16-32-2'
    with open('./_MNISTModels/'+modelName+'_log.pt', 'rb') as infile:
        dict = pickle.load(infile)

    model.load_state_dict(torch.load('./_MNISTModels/'+modelName+'.pt'))
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

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(dict[modelName]['trainLoss'])
    axes[1].plot(dict[modelName]['validationAccuracy'])
    plt.show()
    print("Finish")








