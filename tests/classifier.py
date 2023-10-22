import sys

import matplotlib.pyplot as plt

import torch
from torch import Tensor

import knets as nn

# https://discuss.pytorch.org/t/generating-random-tensors-according-to-the-uniform-distribution-pytorch/53030
x0 = torch.normal(-2, 1, (100, 2))
x1 = torch.normal(2, 1, (100, 2))
y0 = torch.zeros((100, 1), dtype=torch.int32)
y1 = torch.ones((100, 1), dtype=torch.int32)
x = torch.cat((x0, x1), axis=0)
y = torch.cat((y0, y1), axis=0)

class Net(nn.Module):
    def __init__(
            self
    ):
        super().__init__()
        w_init = nn.init.RandomUniform()
        b_init = nn.init.Constant(0.1)

        self.l1 = nn.layers.Dense(2, 10, nn.act.tanh, w_init, b_init)
        self.l2 = nn.layers.Dense(10, 10, nn.act.tanh, w_init, b_init)
        self.out = nn.layers.Dense(10, 1, nn.act.sigmoid)

    def forward(
            self,
            x,
    ):
        x = self.l1(x)
        x = self.l2(x)
        o = self.out(x)
        return o

net = Net()
optimizer = nn.optimizers.Adam(net.params, learning_rate=0.1)
loss_fn = nn.losses.SigmoidCrossEntropy()

for step in range(30):
    o = net.forward(x)
    loss = loss_fn(o, y)
    net.backward(loss)
    optimizer.step()
    acc = nn.metrics.accuracy(o.data > 0.5, y)
    print("Step: %i | loss: %.5f | acc: %.2f" % (step, loss.data, acc))

print(net.forward(x[:10]).data.ravel(), "\n", y[:10].ravel())
plt.scatter(x[:, 0], x[:, 1], c=(o.data > 0.5).ravel(), s=100, lw=0, cmap='RdYlGn')
plt.show()
