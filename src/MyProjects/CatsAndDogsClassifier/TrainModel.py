import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

dataTransform = transforms.Compose([transforms.Resize(size=(128, 128)),
                                    transforms.ToTensor()])

trainDataset= datasets.ImageFolder(root='./Data/training_set/', transform=dataTransform)

trainLoader = torch.utils.data.DataLoader(trainDataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)


x, y = trainDataset[1001]
print(y)
plt.imshow (x[0])
plt.show()