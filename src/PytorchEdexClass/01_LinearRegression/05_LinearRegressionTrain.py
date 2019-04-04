
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

def forward(x):
    y = w*x + b
    return y

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)



class Data(Dataset):

    def __init__(self):

        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.f = 9.0 * self.x - 8
        self.y = self.f + 0.8 * torch.randn(self.x.size())
        self.len = self.x.shape[0]


    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len


dataset = Data()
print("The length of dataset: ", len(dataset))
trainloader = DataLoader(dataset = dataset, batch_size = 30)

w = torch.tensor(-50.0, requires_grad=True)
b = torch.tensor( 30.0,  requires_grad=True)
lr =0.1

LOSS = []
for epoch in range(6):

    Yhat = forward(dataset.x)
    LOSS.append(criterion(Yhat,dataset.y).detach().numpy())
    count =0
    for x,y in trainloader:
        count+=1
        Yhat = forward(x)
        loss = criterion(Yhat,y)
        loss.backward()

        w.data = w.data - lr*w.grad.data
        w.grad.data.zero_()

        b.data = b.data - lr * b.grad.data
        b.grad.data.zero_()

    print(count)
    
print(w, b)
plt.plot(LOSS)
plt.show()