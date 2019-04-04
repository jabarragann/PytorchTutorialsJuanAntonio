
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch


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



if __name__ == "__main__":
    _tasks = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])


    mnist = MNIST("data", download=True, train= True, transform =_tasks)


    split = int(0.8 * len(mnist))
    index_list = list(range(len(mnist)))
    train_idx, valid_idx = index_list[:split], index_list[split:]

    tr_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    trainLoader = DataLoader(mnist, batch_size=256, sampler=tr_sampler)
    validLoader = DataLoader(mnist, batch_size=256, sampler=val_sampler)


    model = Model()


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)


    for epoch in range(1, 11): # run the model for 10 epochs
        train_loss, valid_loss = [], []

        # training part
        model.train()
        for data, target in trainLoader:
            optimizer.zero_grad()

            # 1. forward propagation
            data = data.reshape((-1, 784))
            output = model(data)

            # 2. loss calculation
            loss = loss_function(output, target)

            # 3. backward propagation
            loss.backward()

            # 4. weight optimization
            optimizer.step()

            train_loss.append(loss.item())

        # evaluation part
        model.eval()
        for data, target in validLoader:
            data = data.reshape((-1, 784))
            output = model(data)
            loss = loss_function(output, target)
            valid_loss.append(loss.item())

        print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

    # dataloader for validation dataset
    dataiter = iter(validLoader)
    data, labels = dataiter.__next__()
    output = model(data.reshape((-1, 784)))

    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())

    print("Actual:", labels[:10])
    print("Predicted:", preds[:10])