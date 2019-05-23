import torch.nn as nn

class ConvNetwork(nn.Module):

    def __init__(self,out_1=2,out_2=1,out_3=1):

        super(ConvNetwork,self).__init__()

        #First Convolutional Layer
        self.cnn1 = nn.Conv2d(in_channels=14,out_channels=out_1,kernel_size=1,padding=0)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu1 = nn.ReLU()

        #Second Convolutional Layer
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=3, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        # Third Convolutional Layer
        self.cnn3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=1, padding=0)
        self.relu3 = nn.ReLU()

        #Fully connected Layer
        self.fc1 = nn.Linear(out_3*26*4, 784)
        self.fc2 = nn.Linear(784, 2)

    def forward(self,x):

        out = self.cnn1(x)
        out = self.relu1(out)
        #out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.cnn3(out)
        out = self.relu3(out)

        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

