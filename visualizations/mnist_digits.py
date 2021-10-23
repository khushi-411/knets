import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
        
"""
   Function to load dataset.
"""

def load_data(file_path):
  try:
    df = pd.read_csv(file_path)
    return df

  except FileNotFoundError as error:
    print("File Not Found Error during loading data")
    raise error

  except Exception as error:
      print("Exception Error: load_data")
      print(error)
      return ""

def plot_digit(data, label):
  try:

    fig, axis = plt.subplots(5, 5, figsize = (16, 16)) 
    k = 0 

    for i in range(5):
      for j in range(5): 

        # To plot image 
        axis[i, j].imshow(data[k].reshape(28, 28), interpolation = "none", cmap = "gray")

        # To print label            
        axis[i, j].set_ylabel("label:" + str(label[k].item()))     

        k +=1
    plt.savefig("mnist_digits.png")

  except Exception as error:
    print("Error while plotting digit")
    print(error)

def pie_plot(df_train):
    try:
        counts = df_train.groupby('label')["label"].count()
        label = counts
        count = counts.index
        plt.figure(figsize=(10, 10))
        plt.pie(counts, labels=label)
        plt.legend(count, loc='lower right')
        plt.title('Pie Chart')
        plt.savefig("mnist_pie_chart.png")

    except AttributeError as error:
        print("Attribute Error Occured.")
        print("The error is ", error)

    except ValueError as error:
        print("Value Error Occured.")
        print("The error is ", error)

train_file = "/home/khushi/Documents/simple-neural-network/datasets/data/mnist_train.csv"
test_file = "/home/khushi/Documents/simple-neural-network/datasets/data/mnist_test.csv"

df_train = load_data(train_file)
df_test = load_data(test_file)
train_all = df_train.iloc[:,1:]
train_all_numpy = train_all.to_numpy()
train_label = df_train["label"]
train_label_numpy = train_label.to_numpy()

plot_digit(train_all_numpy, train_label_numpy)
pie_plot(df_train)
