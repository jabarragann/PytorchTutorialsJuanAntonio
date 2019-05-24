import torch
from torch import nn


#convolutional Basics
conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2)
conv.state_dict()['weight'][0][0]=torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0.0,-1.0]])
conv.state_dict()['bias'][0]=1.0
conv.state_dict()

image = torch.zeros(1,1,5,5)
image[0,0,:,2]=1

print(conv(image))

# Multiple inputs multiple output convolutions



#Max pooling

sampleTensor = torch.tensor([[[4,3,-1],[2,-1,1],[0,1,0.5]]])
max3=torch.nn.MaxPool2d(2,stride=1)
print(max3(sampleTensor))

print("example 2")
image1=torch.zeros(1,1,4,4)
image1[0,0,0,:]=torch.tensor([1.0,2.0,3.0,-4.0])
image1[0,0,1,:]=torch.tensor([0.0,2.0,-3.0,0.0])
image1[0,0,2,:]=torch.tensor([0.0,2.0,3.0,1.0])

print(image1)
max3=torch.nn.MaxPool2d(2,stride=1)
print(max3(image1))
max1=torch.nn.MaxPool2d(2)
print(max1(image1))

'''
Very important!!
if the stride parameter is not set the maxpool will do the automatically 
set the maximum possible stride parameter
'''

#Avg2d function example
print("AVG results")
t = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 3.0, 3.0], [3.0, 6.0, 4.0, 5.0], [4.0, 1.0, 2.0, 2.0]])
t = t.view(1, 4, 4)
avg = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
result = avg(t)
print(result)

#zero padding example
conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=2)
conv.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])
conv.state_dict()['bias'][0]=0.0

tensor = torch.tensor([[1.0,2.0],[3.0,4.0]])
tensor = tensor.view(1,1,2,2)
print(conv(tensor))

#Convolutional layers example

out_1=16
out_2=32

cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
relu1 = nn.ReLU()
maxpool1 = nn.MaxPool2d(kernel_size=2)
cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
relu2 = nn.ReLU()
maxpool2 = nn.MaxPool2d(kernel_size=2)
fc1 = nn.Linear(out_2 * 4 * 4, 10)


x = torch.zeros((16,16))
x = x.view(1,1,16,16)
out = cnn1(x)
out = relu1(out)
out = maxpool1(out)
out = cnn2(out)
out = relu2(out)
out = maxpool2(out)
out = out.view(out.size(0), -1)
out = fc1(out)

#Upsample

upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
x = torch.zeros((4,4)) + 1.0 + torch.rand((4,4))*0.5
x = x.view(1,1,4,4)
x_up = upsample(x)