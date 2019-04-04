import torch
import numpy as np
import matplotlib.pyplot as plt

a = torch.FloatTensor([0,1,2,3,4])
print(a)
print(a.dtype)
print(a.type)
print(a.ndimension())


a_col = a.view(-1,1)
print(a)


arr = np.array([1,2.3,45,6.0])

torch_tensor = torch.from_numpy(arr)
back_to_numpy = torch_tensor.numpy()

print("Derivatives Pytorch")

x = torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
print(x)
Y = x**2
y=torch.sum(Y)
y.backward()

print("Gradient",x.grad)
x.grad.zero_()
print("Gradient",x.grad)

x2 = torch.tensor(2.0,requires_grad=True)
y2 = x2**2+2*x2+1
y2.backward()
print("Gradient", x2.grad)
print("Gradient", x2.grad)

print("partial derivatives")

u = torch.tensor(1.0,requires_grad=True)
v = torch.tensor(2.0,requires_grad=True)

f = u*v+u**2
f.backward()
print(v.grad)
print(u.grad)

print("cool hack")

x = torch.linspace(-10,10,10,requires_grad=True)
Y = x**2
y = torch.sum(x**2)
y.backward()


fig,axes = plt.subplots(2,1)

axes[0].plot(x.detach().numpy(),Y.detach().numpy(),label='function')
axes[0].plot(x.detach().numpy(),x.grad.detach().numpy(),label='derivative')
axes[0].legend()

import torch.nn.functional as F
x = torch.linspace(-3,3,1000,requires_grad=True)
Y = F.relu(x)
y = torch.sum(F.relu(x))
y.backward()


axes[1].plot(x.detach().numpy(),Y.detach().numpy(),label='function')
axes[1].plot(x.detach().numpy(),x.grad.detach().numpy(),label='derivative')
axes[1].legend()


plt.show()



