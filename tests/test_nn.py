import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import Tensor

import knets as nn


class Net(nn.Module):
    def __init__(
            self
    ):
        super(Net, self).__init__()
        weights = nn.init.RandomUniform()
        bias = nn.init.Constant(0.1)
        
        """Multi-layer perceptron, here: 2."""
        self.layer_1 = nn.layers.Dense(28*28, 512, nn.act.tanh, weights, bias)
        self.layer_2 = nn.layers.Dense(512, 512, nn.act.tanh, weights, bias)
        self.out = nn.layers.Dense(512, 10, nn.act.sigmoid)
        
        """Single layer perceptron."""
        #self.layer_1 = layers.Dense(28*28, 128, act.tanh, weights, bias)
        #self.out = layers.Dense(128, 10, act.sigmoid)

    def forward(
            self,
            x,
    ):

        """Multi-layer perceptron, here: 2."""
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.out(x)

        """Single layer perceptron."""
        #x = self.layer_1(x)
        #x = self.out(x)

        return x

test = pd.read_csv('tests/datasets/data/mnist_test.csv').astype(np.float32)
test_data = pd.DataFrame(test.iloc[:, 1:])
test_target = pd.DataFrame(test.iloc[:, 0])

test_data = torch.tensor(test_data.values)
test_target = torch.tensor(test_target.values)

"""Normalizing the input pixel/image."""
test_data = torch.nn.functional.normalize(test_data, p=2.0)
test_target = torch.nn.functional.normalize(test_target, p=2.0)

"""create empty data-frame to save results."""
df_loss = pd.DataFrame(columns=["Step", "Value"])
df_acc = pd.DataFrame(columns=["Step", "Value"])
#df_roc = pd.DataFrame(columns=["Step", "Value"])     # binary classification
#df_auc = pd.DataFrame(columns=["Step", "Value"])     # binary classification
df_precision = pd.DataFrame(columns=["Step", "Value"])
df_recall = pd.DataFrame(columns=["Step", "Value"])
df_specificity = pd.DataFrame(columns=["Step", "Value"])
df_f1_score = pd.DataFrame(columns=["Step", "Value"])

net = Net()
optimizer = nn.optimizers.Adam(net.params, learning_rate=0.001)
loss_func = nn.losses.SigmoidCrossEntropy()

"""Testing"""
for step in range(500):
    o = net.forward(test_data)
    loss = loss_func(o, test_target)
    net.backward(loss)
    optimizer.step()
    
    if step%2 == 0:
        loss_dict = {"Step": step, "Value": loss.data.tolist()}
        df_loss = df_loss._append(loss_dict, ignore_index=True)

        acc = nn.metrics.accuracy(o.data > 0.5, test_target)
        acc_dict = {"Step": step, "Value": acc.tolist()}
        df_acc = df_acc._append(acc_dict, ignore_index=True)

        """Use ROC for binary classification."""
        #roc_val = nn.metrics.roc(o.data, test_target, num_thresholds=100)
        #roc_dict = {"Step": step, "Value": roc_val}
        #df_roc = df_roc._append(roc_dict, ignore_index=True)

        """Use AUC for binary classification."""
        #auc_val = nn.metrics.auc(o.data, test_target, num_thresholds=100)
        #auc_dict = {"Step": step, "Value": auc_val.tolist()}
        #df_auc = df_auc._append(auc_dict, ignore_index=True)

        precision_val = nn.metrics.precision(o.data.to(torch.int32), test_target.to(torch.int32))
        pre_dict = {"Step": step, "Value": precision_val.tolist()}
        df_precision = df_precision._append(pre_dict, ignore_index=True)

        recall_val = nn.metrics.recall(o.data.to(torch.int32), test_target.to(torch.int32))
        recall_dict = {"Step": step, "Value": recall_val.tolist()}
        df_recall = df_recall._append(recall_dict, ignore_index=True)

        spe = nn.metrics.recall(o.data.to(torch.int32), test_target.to(torch.int32))
        spe_dict = {"Step": step, "Value": spe.tolist()}
        df_specificity = df_specificity._append(spe_dict, ignore_index=True)

        f1 = nn.metrics.recall(o.data.to(torch.int32), test_target.to(torch.int32))
        f1_dict = {"Step": step, "Value": f1.tolist()}
        df_f1_score = df_f1_score._append(f1_dict, ignore_index=True)

        print("Step: %i | loss: %.5f | acc: %.5f | precision: %.5f | recall: %.5f | specificity: %.5f | f1_score: %.5f" % (step, loss.data, acc, precision_val, recall_val, spe, f1))

df_loss.to_csv("test_loss.csv", index=False, header=True)
df_acc.to_csv("test_accuracy.csv", index=False, header=True)
#df_roc.to_csv("test_roc.csv", index=False, header=True)
#df_auc.to_csv("test_auc.csv", index=False, header=True)
df_precision.to_csv("test_precision.csv", index=False, header=True)
df_recall.to_csv("test_recall.csv", index=False, header=True)
df_specificity.to_csv("test_specificity.csv", index=False, header=True)
df_f1_score.to_csv("test_f1_score.csv", index=False, header=True)
