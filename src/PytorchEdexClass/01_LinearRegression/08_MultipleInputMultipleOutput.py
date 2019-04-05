import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from itertools import product

class LR(nn.Module):

    def __init__(self, in_features=1,out=1):
        super(LR, self).__init__()
        torch.manual_seed(1)
        self.linear = nn.Linear(in_features, out)

    def forward(self, x):
        out = self.linear(x)

        return out

class Data(Dataset):

    def __init__(self):
        #Features along a line

        # self.x = torch.zeros(40,2)
        # self.x[:,0] = torch.linspace(-3,3,steps=40)
        # self.x[:,1] = torch.linspace( -3,3,steps =40)

        #Features forming a grid in 2d space
        self.x= np.linspace(-3, 3, num=8,dtype='float32')
        self.x = torch.from_numpy(np.array(list((product(self.x, repeat=2)))))

        self.w = torch.tensor([[10.0, -10],[3,-3.5]])
        self.b = torch.tensor([[15.0, 6]])

        self.f = torch.mm(self.x,self.w) + self.b

        self.y = self.f + 0.1*torch.randn((self.x.shape))
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":

    model = LR(in_features=2, out=2)

    myData = Data()
    criterion = nn.MSELoss()
    trainingData = DataLoader(dataset= myData, batch_size = 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainingLoss = []
    for epoch in range(100):

        for xt,yt in trainingData:

            #It is absolutely indispensable to set the optimizer gradients to zero everytime!
            #If you do not do this, the training of the model will not work.
            optimizer.zero_grad()
            yhat = model(xt)
            loss = criterion(yhat,yt)
            loss.backward()
            optimizer.step()

        trainingLoss.append(loss.detach().numpy())

    for params in model.parameters():
        print(params)
        print(params.shape)

    x,y = myData[[3,8,29]]

    w = torch.tensor([[10.0, -10],[3,-3.5]])
    b = torch.tensor([[15.0, 6]])
    f = torch.mm(x, w) + b

    y_predict = model(x)

    print("\nTest values")
    for xx, g1,h1 in zip(x,y,y_predict):
        print("point: ({:+07.3f},{:+07.3f}) label: ({:+07.3f},{:+07.3f}) predicted: ({:+07.3f},{:+07.3f})".format(\
                                                                                 xx[0],xx[1],g1[0],g1[1],h1[0],h1[1]))

    print("Final Loss function: {:0.3f}".format(float(trainingLoss[-1])) )
    plt.plot(trainingLoss)
    plt.xlabel('epochs')
    plt.ylabel('MSE loss function')
    plt.show()

    print("finish!")
