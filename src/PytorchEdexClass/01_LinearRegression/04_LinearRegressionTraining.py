import torch
import matplotlib.pyplot as plt
import numpy as np



def forward(x):
    y = w*x + b
    return y

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)


w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-3.0,  requires_grad=True)

X = torch.arange(-3,3,0.1).view(-1,1)
f = 9.0*X - 8
Y = f + 0.8*torch.randn(X.size())

# plt.plot(X.numpy(),f.numpy())
# plt.plot(X.numpy(),Y.numpy(),'ro')
# plt.show()


lr= 0.1

LOSS = []
for epoch in range(6):

    Yhat = forward(X)
    loss = criterion(Yhat,Y)
    loss.backward()

    w.data = w.data - lr*w.grad.data
    w.grad.data.zero_()

    b.data = b.data - lr * b.grad.data
    b.grad.data.zero_()

    temp = loss.detach().numpy()
    LOSS.append(temp)


print(LOSS)
plt.plot(LOSS)
plt.show()

print(w,b)
plt.plot(X.numpy(),f.numpy())
plt.plot(X.numpy(),forward(X).detach().numpy(),'ro')
plt.show()


