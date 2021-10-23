import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import Tensor

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
    def __init__(
            self
    ) -> None:

        super(Net, self).__init__()
        """Other ways for initialization: init.RandomNormal(), init.TruncatedNormal()"""
        weights = init.RandomUniform()
        """Other ways bias can be initialize: init.Zeros(), init.Ones()"""
        bias = init.Constant(0.1)
        
        """Layers; 
        Parameters are:
            input_channel,
            output_channel,
            activation,
            w_initializer,
            b_initializer,
            use_bias
        Returns:
        """
        """Other activation function that can be used: act.Linear, act.ReLU, act.LeakyReLU, act.ELU, act.Sigmoid, act.SoftPlus, act.SoftMax"""
        # TODO: create two function to design layers; 1st function is used to create layers and the other is used to apply activation function.
        
        """Multi-layer perceptron, here: 2."""
        self.layer_1 = layers.Dense(28*28, 512, act.tanh, weights, bias)
        self.layer_2 = layers.Dense(512, 512, act.tanh, weights, bias)
        self.out = layers.Dense(512, 10, act.sigmoid)

        """Single layer perceptron."""
        #self.layer_1 = layers.Dense(28*28, 128, act.tanh, weights, bias)
        #self.out = layers.Dense(128, 10, act.sigmoid)

    def forward(
            self,
            x: Tensor
    ) -> variable.Variable:
        
        """Multi-layer perceptron, here: 2."""
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.out(x)

        """Single layer perceptron."""
        #x = self.layer_1(x)
        #x = self.out(x)

        return x

"""Load dataset"""
train = pd.read_csv('/home/khushi/Documents/simple-neural-network/datasets/data/mnist_train.csv').astype(np.float32)
train_data = pd.DataFrame(train.iloc[:, 1:])
train_target = pd.DataFrame(train.iloc[:, 0])

"""Converting into tensors."""
train_data = torch.tensor(train_data.values)
train_target = torch.tensor(train_target.values)

"""Normalizing the input pixel/image."""
train_data = torch.nn.functional.normalize(train_data, p=2.0)
train_target = torch.nn.functional.normalize(train_target, p=2.0)

"""create empty data-frame to save results."""
df_loss = pd.DataFrame(columns=["Step", "Value"])
df_acc = pd.DataFrame(columns=["Step", "Value"])
#df_roc = pd.DataFrame(columns=["Step", "Value"])     # binary classification
#df_auc = pd.DataFrame(columns=["Step", "Value"])     # binary classification
df_precision = pd.DataFrame(columns=["Step", "Value"])
df_recall = pd.DataFrame(columns=["Step", "Value"])
df_specificity = pd.DataFrame(columns=["Step", "Value"])
df_f1_score = pd.DataFrame(columns=["Step", "Value"])

"""Network class instantiation"""
net = Net()

"""Other optimizers that can be used: optimizers.SGD(), optimizers.Momentum(), optimizers.AdaGrad(), optimizers.Adadelta(), optimizers.RMSProp(), optimizers.Adam(), optimizers.AdaMax()"""
optimizer = optimizers.Adam(net.params, learning_rate=0.001)

"""Other loss functions that can be used: losses.MSE(), losses.CrossEntropy(), losses.SoftMaxCrossEntropy(), losses.SoftMaxCrossEntropyWithLogits(), losses.SparseSoftMaxCrossEntropy(), losses.SparseSoftMaxCrossEntropyWithLogits()"""
loss_func = losses.SigmoidCrossEntropy()

"""Training"""
for step in range(500):
    o = net.forward(train_data)
    #k = o.data > 0.7
    #print(k.to(torch.int32))
    loss = loss_func(o, train_target)
    net.backward(loss)
    optimizer.step()

    if step%2 == 0:
        loss_dict = {"Step": step, "Value": loss.data.tolist()}
        df_loss = df_loss.append(loss_dict, ignore_index=True)

        acc = metrics.accuracy(o.data > 0.5, train_target)
        acc_dict = {"Step": step, "Value": acc.tolist()}
        df_acc = df_acc.append(acc_dict, ignore_index=True)

        """Use ROC for binary classification."""
        #roc_val = metrics.roc(o.data, train_target, num_thresholds=100)
        #roc_dict = {"Step": step, "Value": roc_val}
        #df_roc = df_roc.append(roc_dict, ignore_index=True)

        """Use AUC for binary classification."""
        #auc_val = metrics.auc(o.data, train_target, num_thresholds=100)
        #auc_dict = {"Step": step, "Value": auc_val.tolist()}
        #df_auc = df_auc.append(auc_dict, ignore_index=True)

        precision_val = metrics.precision(o.data.to(torch.int32), train_target.to(torch.int32))
        pre_dict = {"Step": step, "Value": precision_val.tolist()}
        df_precision = df_precision.append(pre_dict, ignore_index=True)
        
        recall_val = metrics.recall(o.data.to(torch.int32), train_target.to(torch.int32))
        recall_dict = {"Step": step, "Value": recall_val.tolist()}
        df_recall = df_recall.append(recall_dict, ignore_index=True)

        spe = metrics.recall(o.data.to(torch.int32), train_target.to(torch.int32))
        spe_dict = {"Step": step, "Value": spe.tolist()}
        df_specificity = df_specificity.append(spe_dict, ignore_index=True)
        
        f1 = metrics.recall(o.data.to(torch.int32), train_target.to(torch.int32))
        f1_dict = {"Step": step, "Value": f1.tolist()}
        df_f1_score = df_f1_score.append(f1_dict, ignore_index=True)
        
        print("Step: %i | loss: %.5f | acc: %.5f | precision: %.5f | recall: %.5f | specificity: %.5f | f1_score: %.5f" % (step, loss.data, acc, precision_val, recall_val, spe, f1))

df_loss.to_csv("loss.csv", index=False, header=True)
df_acc.to_csv("accuracy.csv", index=False, header=True)
#df_roc.to_csv("roc.csv", index=False, header=True)
#df_auc.to_csv("auc.csv", index=False, header=True)
df_precision.to_csv("precision.csv", index=False, header=True)
df_recall.to_csv("recall.csv", index=False, header=True)
df_specificity.to_csv("specificity.csv", index=False, header=True)
df_f1_score.to_csv("f1_score.csv", index=False, header=True)
