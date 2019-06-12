import torch.nn as nn

class ConvNetwork(nn.Module):

    def __init__(self, out_1=2, out_2=1, out_3=12, activation="relu"):

        super(ConvNetwork,self).__init__()

        #SET activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'elu':
            self.activation = nn.ELU()

        #First Convolutional Layer
        self.cnn1 = nn.Conv2d(in_channels=14,out_channels=out_1,kernel_size=1,padding=0)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn1 = nn.BatchNorm2d(out_1)
        self.relu1 = self.activation

        #Second Convolutional Layer
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=3, padding=2)
        self.bn2 = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = self.activation

        # Third Convolutional Layer
        self.cnn3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_3)
        self.relu3 = self.activation

        #Fully connected Layer
        self.fc1 = nn.Linear(out_3*26*4, 784)
        self.fc_bn = nn.BatchNorm1d(784)
        self.fc2 = nn.Linear(784, 2)
        self.relu4 = self.activation

    def forward(self,x):

        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        #out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.maxpool2(out)
        out = self.relu2(out)

        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc_bn(out)
        out = self.relu4(out)

        out = self.fc2(out)

        return out

