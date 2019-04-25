import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import time, pickle
import CNN1 as cnn

# GLOBAL VARIABLES
IMAGE_SIZE = 64
CONV_LAYER1 = 32
CONV_LAYER2 = 16
DATASET_PATH = "./Data/"
DEVICE = 'cpu'


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename,map_location=DEVICE)
    epoch = checkpoint_dict['epoch']
    trLossH = checkpoint_dict['trLossH']
    valLossH = checkpoint_dict['valLossH']
    valAccH = checkpoint_dict['valAccH']

    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    return epoch, trLossH, valLossH, valAccH

if __name__ == "__main__":
    # 1 Create Model and Dataset
    dataTransform = transforms.Compose([transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
                                        transforms.ToTensor()])
    trainData = datasets.ImageFolder(root=DATASET_PATH + '/training_set/', transform=dataTransform)
    testData = datasets.ImageFolder(root=DATASET_PATH + '/test_set/', transform=dataTransform)

    model = cnn.ConvNetwork(out_1=CONV_LAYER1, out_2=CONV_LAYER2)
    model.to(DEVICE)

    # 2 Sampler and Data loader
    split = int(0.8 * len(trainData))
    idxArr = np.random.permutation(len(trainData))
    trainIdx, valIdx = idxArr[:split], idxArr[split:]
    trainSampler = SubsetRandomSampler(trainIdx)
    valSampler = SubsetRandomSampler(valIdx)

    trainLoader = DataLoader(trainData, batch_size=256, sampler=trainSampler)
    validLoader = DataLoader(trainData, batch_size=256, sampler=valSampler)
    testLoader = DataLoader(testData, batch_size=256, shuffle=True)

    # 3 Optimizer
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

    trainLogger = {}
    modelName = 'cnn16-32-64_v2'
    trainLogger[modelName] = {'CONV_LAYER1': CONV_LAYER1, 'CONV_LAYER2': CONV_LAYER2, 'IMAGE_SIZE': IMAGE_SIZE}

    valAccuracyHistory = []
    valLossHistory = []
    trLossHistory = []
    trLossHistoryPerBatch = []

    # print("Test Loaders ...")
    #
    # # TEST Loaders
    # fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    # trainLoaderIter = iter(validLoader)
    # for i in range(5):
    #     x, y = trainLoaderIter.next()
    #     x = x[0]
    #     y = y[0]
    #     # yHat = model(x)
    #     # loss = lossFunction(yHat, y)
    #     # loss.backward()
    #     # optimizer.step()
    #     axes[i].imshow(np.transpose(x.numpy(), (1, 2, 0)))
    #     axes[i].set_title('cat' if y.numpy() == np.array([0]) else 'dog')

    # 5 test Model
    loadFile = './Model1/classifier-{:03d}.pkl'.format(27)
    load = True
    # Loading model parameters
    if load:
        first_epoch, trLossHistory, valLossHistory, valAccuracyHistory = load_checkpoint(optimizer, model, loadFile)
        print("Loading model from epoch {:3d}".format(first_epoch))

    testLoader = DataLoader(testData, batch_size=256, shuffle=True)
    print("test model...")

    correct = 0
    total = 0
    model.eval()
    for x, y in testLoader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        z = model(x)
        yHat = z.argmax(1)
        correct += (yHat == y).sum().item()
        total += y.shape[0]

    accuracy = correct / total

    print("Model Accuracy on test set: {:d}/{:d} ({:.4f})".format(correct, total, accuracy))
