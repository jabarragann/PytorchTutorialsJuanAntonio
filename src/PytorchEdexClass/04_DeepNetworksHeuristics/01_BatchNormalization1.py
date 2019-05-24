import torch.nn as nn
import torch

# With Learnable Parameters
m1 = nn.BatchNorm1d(10)
# Without Learnable Parameters
m2 = nn.BatchNorm1d(10, affine=False)

input = torch.randn(20, 10)
output = m1(input)

params = m1._parameters
mean   = m1.running_mean
var    = m1.running_var


#Example Linear Example
linear = nn.Linear(3, 2, bias=True)
weight = torch.tensor([[1.0,2.0,3.0],[2.0,1.0,0.0]])
bias = torch.tensor([0.0,0.0])
x = torch.tensor([[0.0,1.0,0.0],[1.0,2.0,4.0],[-2.0,5.0,6.0]])

bn = nn.BatchNorm1d(2)
bn._parameters['weight'] = torch.tensor([1.0,1.0])

linear._parameters['weight'] = weight
linear._parameters['bias'] = bias

y = linear(x)
z = bn(y)


#Convolutional example
x2d = torch.tensor([[[[1.0,5.0,8.0,4.0],[2.0,6.0,7.0,3.0],[3.0,7.0,6.0,2.0],[4.0,8.0,5.0,1.0]]]])
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride =1)
conv_bn = nn.BatchNorm2d(1)

conv._parameters['weight'][0][0] = torch.tensor([[1.0,1.0],[1.0,1.0]])
conv._parameters['bias'][0] = 0.0

conv_bn._parameters['weight'][0] = 4.618
conv_bn._parameters['bias'][0] = 20.66

y2d = conv(x2d)
z2d = conv_bn(y2d)