
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torch.nn import Linear
import torch.nn as nn


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


class LR(nn.Module):

    def __init__(self,in_size,out_size):
        #nn.Module.__init__(self)
        super(LR, self).__init__()
        torch.manual_seed(1)
        self.linear = nn.Linear(in_size,out_size)

    def forward(self,x):
        out= self.linear(x)
        return out



dataset = Data()
criterion = nn.MSELoss()
trainloader = DataLoader(dataset=dataset,batch_size=10)

model = LR(1,1)
optimizer = torch.optim.SGD(model.parameters(),lr= 0.005)


LOSS = []
for epoch in range(100):

    for x,y in trainloader:

        yhat = model(x)
        loss = criterion(yhat,y)
        optimizer.zero_grad()
        LOSS.append(loss.detach().numpy())

        loss.backward()
        optimizer.step()


torch.save(model.state_dict(),'best_model.pt')
print(LOSS[0],LOSS[-1])
print('Parameters: ', list(model.parameters()))
plt.plot(LOSS)
plt.show()

#To load the model
# model_best = linear_regression(1,1)
# model_best.load_state_dict(torch.load('best_model.pt'))
# yhat = model_best(val_data.x)