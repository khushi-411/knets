import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/datasets')
import mnist
from dataloader import DataLoader

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/models')
import module as nn
import layers

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/ops')
import initializers as init, activations as act, losses, optimizers, variable

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/eval')
import metrics

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        weights = init.RandomUniform()
        bias = init.Constant(0.1)

        self.layer_1 = layers.Dense(1, 1000, act.tanh, weights, bias)
        self.layer_2 = layers.Dense(1000, 3000, act.tanh, weights, bias)
        self.layer_3 = layers.Dense(3000, 10000, act.tanh, weights, bias)
        self.layer_4 = layers.Dense(10000, 5000, act.tanh, weights, bias)
        self.layer_5 = layers.Dense(5000, 100, act.tanh, weights, bias)
        self.out = layers.Dense(100, 10, act.sigmoid)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.out(x)
        return x

train_data = mnist.MNIST(
        root='/home/khushi/Documents/simple-neural-network/datasets',
        train=True,
        transform=transforms.ToTensor()
)

train_data_loader = DataLoader(
        train_data[0],
        train_data[1],
        batch_size=64
)

net = Net()
optimizer = optimizers.Adam(net.params, learning_rate=0.1)
loss_func = losses.SigmoidCrossEntropy()

for step in range(10):
    print(step)
    bx, by = train_data_loader.next_batch()
    print(bx)
    by_ = net.forward(bx)
    loss = loss_func(by_, by)
    net.backward(loss)
    optimizer.step()
    if step % 5 == 0:
        ty_ = net.forward(test_x)
        acc = metrics.accuracy(torch.argmax(ty_.data, axis=1), test_y)
        print("Step: %i | loss: %.3f | acc: %.2f" % (step, loss.data, acc))
