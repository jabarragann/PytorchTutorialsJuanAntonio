import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

#Simple Example
print("Simple example")

def forward(x):
    y=torch.mm(x,w)+b

    return y

w = torch.tensor([[2.0],[3.0]],requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)


x = torch.tensor([[1.0,1.0],[4.0,-2.0],[1.0,3.0]])

yhat = forward(x)

print(yhat)

#Complicated example training a 2d linear regression

print("Training a 2d linear regression")

class Data(Dataset):

    def __init__(self):

        self.x = torch.zeros(20,2)
        self.x[:,0] = torch.arange(-1,1,0.1)
        self.x[:,1] = torch.arange(-1,1,0.1)
        self.b = torch.tensor(15.0)
        self.w = torch.tensor([[1.0],[1.0]])

        self.f = torch.mm(self.x,self.w) + self.b
        self.y = self.f + 0.1*torch.randn((self.x.shape))
        self.len = self.x.shape[0]


    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len


class LR(nn.Module):

    def __init__(self, in_features=1,out=1):
        # nn.Module.__init__(self)
        super(LR, self).__init__()
        torch.manual_seed(1)
        self.linear = nn.Linear(in_features, out)

    def forward(self, x):
        out = self.linear(x)

        return out


data_set = Data()
criterion = nn.MSELoss()
trainloader = DataLoader(dataset=data_set,batch_size=5)
model =LR(in_features=2, out=1)


optimizer = torch.optim.SGD(model.parameters(),lr=0.1)


LossArray = []
for epoch in range(100):
    for x,y in trainloader:
        yhat = model(x)
        loss =criterion(yhat,y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        LossArray.append(loss.detach().numpy())



print(list(model.parameters()))

plt.plot(LossArray)
plt.show()

print('Finish!')
