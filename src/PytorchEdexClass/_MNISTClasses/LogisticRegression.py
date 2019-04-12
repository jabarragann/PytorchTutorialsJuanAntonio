from torch import nn
import torch

class LogisticRegression(nn.Module):

    def __init__(self, in_features, out):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(in_features, out)

    def forward(self, x):
        x = self.linear(x)
        out = torch.sigmoid(x)
        return out