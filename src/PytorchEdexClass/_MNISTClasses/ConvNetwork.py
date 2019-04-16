import torch
import torch.nn as nn

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