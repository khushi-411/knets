import sys

import numpy as np
import pandas as pd

import torch
from torch import Tensor
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/models')
import module as nn
import layers

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/ops')
import initializers as init, activations as act, losses, optimizers, variable

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/eval')
import metrics

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/datasets')
import dataloader

class CNN(nn.Module):

    def __init__(
            self
    ):
        super().__init__()

        self.seq_layers = self.sequential(
                layers.Conv2D(3, 32, kernel_size=3, strides=1, padding='valid', activation=act.relu),
                layers.MaxPool2D(pool_size=3, strides=2, padding='valid'),
                layers.Conv2D(32, 64, kernel_size=3, strides=1, padding='valid', activation=act.relu),
                layers.MaxPool2D(pool_size=3, strides=2, padding='valid'),
                layers.Conv2D(64, 64, kernel_size=3, strides=1, padding='valid', activation=act.relu),
                layers.MaxPool2D(pool_size=3, strides=2, padding='valid'),
                layers.Conv2D(64, 128, kernel_size=3, strides=1, padding='valid', activation=act.relu),
                layers.MaxPool2D(pool_size=3, strides=2, padding='valid'),
                layers.Conv2D(128, 128, kernel_size=3, strides=1, padding='valid', activation=act.relu),
                layers.MaxPool2D(pool_size=3, strides=2, padding='valid'),
                layers.AvgPool2D((1,1))
        )

        self.dnnModel = self.sequential(
                layers.Dense(128, 64, activation=act.tanh),
                layers.Dense(64, 32, activation=act.tanh),
                layers.Dense(32, 8, activation=act.tanh)
        )

        def forward(
                self,
                x
        ):
            output = self.seq_layers.forward(x)
            # output = output.squeeze()
            output = self.dnnModel.forward(x)
            return output

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(root = '/home/khushi/Documents/simple-neural-network/datasets/data/archive/natural_images', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)

cnn = CNN()
opt = optimizers.Adam(cnn.params, learning_rate=0.001)
loss = losses.SparseSoftMaxCrossEntropyWithLogits()

for epoch in range(100):
    for i, (images, target) in enumerate(dataloader):
        output = cnn(images)
        _loss = loss(output, target)
        opt.zero_grad()
        _loss.backward()
        opt.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, 100, i+1, len(dataloader), _loss.item()))
