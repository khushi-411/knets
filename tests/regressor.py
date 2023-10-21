import sys

import matplotlib.pyplot as plt

import torch
from torch import Tensor

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/models')
import module as nn
import layers

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/ops')
import initializers as init, activations as act, losses, optimizers, variable

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/eval')
import metrics

x = torch.linspace(-1, 1, 200)[:, None]       # [batch, 1]
y = x ** 2 + torch.normal(0., 0.1, (200, 1))     # [batch, 1]

class Net(nn.Module):
    def __init__(
            self
    ) -> None:
        super().__init__()
        w_init = init.RandomUniform()
        b_init = init.Constant(0.1)

        self.l1 = layers.Dense(1, 10, act.tanh, w_init, b_init)
        self.l2 = layers.Dense(10, 10, act.tanh, w_init, b_init)
        self.out = layers.Dense(10, 1, act.sigmoid)

    def forward(
            self,
            x: Tensor
    ) -> variable.Variable:
        x = self.l1(x)
        x = self.l2(x)
        o = self.out(x)
        return o

net = Net()
optimizer = optimizers.Adam(net.params, learning_rate=0.1)
loss_fn = losses.SigmoidCrossEntropy()

for step in range(30):
    o = net.forward(x)
    loss = loss_fn(o, y)
    net.backward(loss)
    optimizer.step()
    acc = metrics.accuracy(o.data > 0.5, y)
    print("Step: %i | loss: %.5f | acc: %.2f" % (step, loss.data, acc))

plt.scatter(x, y, s=20)
plt.plot(x, o.data, c="red", lw=3)
plt.show()
