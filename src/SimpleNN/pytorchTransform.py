

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784,128)
        self.output = nn.Linear(128,10)

    def forward(self, x):
        x = self.hidden(x)
        x = F.sigmoid(x)
        x = self.output(x)
        return x




_tasks = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
        ])

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=_tasks)
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=_tasks)



print(len(mnist_trainset))
print(len(mnist_testset))


train_image_zero, train_target_zero = mnist_trainset[0]

train_image = np.array(train_image_zero)

# train_image_zero.resize((280,280)).show()
# print(train_target_zero)

split = int(0.8 * len(mnist_trainset))
index_list = list(range(len(mnist_trainset)))
train_idx, valid_idx = index_list[:split], index_list[split:]

tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

trainLoader = DataLoader(mnist_trainset, batch_size=256, sampler=tr_sampler)
validLoader = DataLoader(mnist_trainset, batch_size=256, sampler=val_sampler)

model = Model()

for data, target in trainLoader:
    print("Hello")
    print(data.shape)
    if data.shape[0] == 128:
        print("error!!!!!!!!!!!!")
    else:
        data = data.reshape((256,784))
        out = model(data)
    print(data)
