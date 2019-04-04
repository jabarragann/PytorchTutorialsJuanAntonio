
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class toy_set(Dataset):

    def __init__(self,length=100,transform =None):
        self.x = 2*torch.ones(length,2)
        self.y = torch.ones(length,1)

        self.len = length
        self.transform = transform


    def __getitem__(self, index):

        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.len

class add_mult(object):
    def __init__(self, addx=1,muly=1):
        self.addx = addx
        self.muly = muly

    def __call__(self,sample):
        x = sample[0]
        y = sample[1]

        x = x+self.addx
        y = y*self.muly
        sample = x,y
        return sample

class mult(object):
    def __init__(self,muly=5):
        self.muly = muly

    def __call__(self,sample):
        x = sample[0]
        y = sample[1]

        x = x*self.muly
        y = y*self.muly
        sample = x,y
        return sample

dataset = toy_set()

for i in range(3):
    print(dataset[i])

a_m = add_mult()
a_m2 = mult()

data_transform = transforms.Compose([a_m,a_m2])
print(a_m(dataset[0]))

dataset2 = toy_set(transform = data_transform)

print("Transformed dataset ")
for i in range(3):
    print(dataset2[i])






