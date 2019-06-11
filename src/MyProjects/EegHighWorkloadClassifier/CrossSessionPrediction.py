import torch
import torch.nn as nn
import EegDataset
import EegConvNetwork
import torch.optim as optim
from torch.utils.data import DataLoader

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


DEVICE = 'cpu'
CONV_LAYER1 = 32
CONV_LAYER2 = 64
CONV_LAYER3 = 12
MODEL = 1


def makePredictions(dataloader, model):
    correct = 0
    total = 0
    for x, y in dataloader:
        z = model(x)
        yHat = z.argmax(1)
        correct += (yHat == y).sum().item()
        total += y.shape[0]

    return correct, total

if __name__ == '__main__':
    d3_data = EegDataset.EegDataset(datasetPath='./Dataset/D3')
    d4_data = EegDataset.EegDataset(datasetPath='./Dataset/D4')
    d3_model = EegConvNetwork.ConvNetwork(out_1=CONV_LAYER1, out_2=CONV_LAYER2, out_3=CONV_LAYER3)
    d4_model = EegConvNetwork.ConvNetwork(out_1=CONV_LAYER1, out_2=CONV_LAYER2, out_3=CONV_LAYER3)

    load_checkpoint(None, d3_model, './Models/CrossSessionTest/D3_classifier.pkl')
    load_checkpoint(None, d4_model, './Models/CrossSessionTest/D4_classifier.pkl')

    d3_loader = DataLoader(d3_data, batch_size=16, shuffle = False)
    d4_loader = DataLoader(d4_data, batch_size=16, shuffle = False)

    print('Test Model 3 on dataset 4')
    correct, total = makePredictions(d3_loader, d4_model)
    print("Accuracy: {:0.5f}".format(correct/total))

    print('Test model 4 on dataset 3 ')
    correct, total = makePredictions(d4_loader, d3_model)
    print("Accuracy: {:0.5f}".format(correct/total))