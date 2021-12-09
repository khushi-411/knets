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

#sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/datasets')
#import dataloader

class CNN(nn.Module):

    def __init__(
            self
    ):
        super().__init__()
        """
        self.seq_layers = self.sequential(
                layers.Conv2D(1, 6, (5, 5), (1, 1), "same", channels_last=True, activation=act.relu),  # [n. 28, 28, 6]
                layers.MaxPool2D(2, 2),  # [n, 14, 14, 6]
                layers.Conv2D(6, 16, 5, 1, "same", channels_last=True, activation=act.relu),  # [n, 14, 14, 16]
                layers.MaxPool2D(2, 2),  # [n, 7, 7, 16]
                layers.Flatten(),  # [n, 7*7*16]
                layers.Dense(7 * 7 * 16, 10, activation=act.relu)
        )
        """

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
                torch.nn.Linear(128, 64),
                torch.nn.Linear(64, 32),
                torch.nn.Linear(32, 8)
        )

        def forward(
                self,
                x
        ):
            output = self.seq_layers(x)
            output = output.squeeze()
            output = self.dnnModel(output)
            return output

        """
        def forward(
                self,
                x
        ):
            o = self.seq_layers.forward(x)
            return o
        """

"""
# Load dataset
train = pd.read_csv('/home/khushi/Documents/simple-neural-network/datasets/data/mnist_train.csv').astype(np.float32)
train_data = pd.DataFrame(train.iloc[:, 1:])
train_target = pd.DataFrame(train.iloc[:, 0])

# Converting into tensors.
train_data = torch.tensor(train_data.values)
train_target = torch.tensor(train_target.values)

# Normalizing the input pixel/image.
train_data = torch.nn.functional.normalize(train_data, p=2.0)
train_target = torch.nn.functional.normalize(train_target, p=2.0)

train_loader = dataloader.DataLoader(train_data, train_target, batch_size=64)
"""

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(root = '/home/khushi/Documents/simple-neural-network/datasets/data/archive/natural_images', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32) #, shuffle = True)

cnn = CNN()
opt = optimizers.Adam(cnn.params, learning_rate=0.001)
loss = losses.SparseSoftMaxCrossEntropyWithLogits()

for step in range(100):
    bx, by = dataloader.next_batch()
    by_ = cnn.forward(bx)
    _loss = loss(by_, by)
    cnn.backward(_loss)
    opt.step()
    if step % 5 == 0:
        ty_ = cnn.forward(train_data)
        acc = metrics.accuracy(torch.argmax(ty_.data, axis=1), train_target)
        print("Step: %i | loss: %.3f | acc: %.2f" % (step, _loss.data, acc))
