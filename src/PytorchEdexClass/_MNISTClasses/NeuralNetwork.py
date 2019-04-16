import torch
from torch import nn

class NeuralNetwork (nn.Module):
    def __init__(self, layers, activation):
        super().__init__()
        self.hidden = nn.ModuleList()
        self.activation = activation
        for inputSize, outputSize in zip(layers,layers[1:]):
            self.hidden.append(nn.Linear(inputSize,outputSize))

    def forward(self, x):

        L = len(self.hidden)
        for(l,linearTransform) in zip(range(L),self.hidden):
            if l<L-1:
                if self.activation  == 'relu':
                    x = torch.nn.functional.relu(linearTransform(x))
                elif self.activation == 'sigmoid':
                    x = torch.sigmoid(linearTransform(x))
                elif self.activation == 'tanh':
                    x = torch.tanh(linearTransform(x))
            else:
                x = linearTransform(x)
        return x