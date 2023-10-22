import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import csv

fig, ax = plt.subplots()
figure(figsize=(70, 40))

x = []
y = []

with open('tests/datasets/results/iteration-10/loss.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    k = 0
    for row in lines:
        if k >= 1 and k % 10 == 0:
            x.append(float(row[0]))
            y.append(float(row[1]))
        k += 1
#print(y)

y1 = []
with open('tests/datasets/results/iteration-11/loss.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    k = 0
    for row in lines:
        if k >= 1 and k % 10 == 0:
            y1.append(float(row[1]))
        k += 1
#print(y1)

ax.plot(x, y, color = 'r', linestyle = 'dashed',
         marker = 'o', label = "Training Loss")
ax.set_xlabel('Number of Steps.')
ax.set_ylabel('Optimizer = SGD; Loss Function = SigmoidCrossEntropy')

ax2 = ax.twinx()

ax2.plot(x, y1, color = 'g', linestyle = 'dotted',
         marker = '+', label = "Training Loss")
ax2.set_ylabel('Optimizer = Adam; Loss Function = MSE')

fig.savefig('norm_loss_iter_10_11.png', format='png')
plt.show()
