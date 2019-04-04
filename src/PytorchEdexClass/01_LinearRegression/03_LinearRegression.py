import torch
from torch.nn import Linear

w = torch.tensor(2.0,requires_grad=True)
b = torch.tensor(-1.0,requires_grad=True)


def forward(x):
    y = w*x + b

    return y

x = torch.tensor([[1.0],[2.0]])

yhat = forward(x)


print(yhat)


#Built-in Linear class
torch.manual_seed(1)
model = Linear(in_features=1,out_features=1)
print(list(model.parameters()))

x = torch.tensor([[1.0],[2.0]])
yhat = model(x)

print(yhat)

#Custom modules
import torch.nn as nn

class LR(nn.Module):

    def __init__(self,in_size,out_size):
        #nn.Module.__init__(self)
        super(LR, self).__init__()
        torch.manual_seed(1)
        self.linear = nn.Linear(in_size,out_size)

    def forward(self,x):
        out= self.linear(x)
        return out

model = LR(1,1)
print(list(model.parameters()))

x = torch.tensor([[1.0],[2.0]])

yhat = model(x)
print(yhat)


