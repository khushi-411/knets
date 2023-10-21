# Simple Neural Network

I aim to build a simple Neural Network from scratch based on [PyTorch](https://pytorch.org/docs/stable/index.html). The sole objective is to understand the fundamentals of deep learning and to understand the backend implementation of the code.

### Usage
You need to install a few libraries to execute these codes. The dependencies are mentioned in [requirements.txt](https://github.com/khushi-411/simple-neural-network/blob/main/requirements.txt) and [environment.yml](https://github.com/khushi-411/simple-neural-network/blob/main/environment.yml). Please take a look!

#### Install
```python
pip install knets
```

#### From source
To download files:

1. Clone the repository
   ```python
   git clone https://github.com/khushi-411/knets.git
   cd knets
   ```
2. Create a virtual Conda environment:
   ```python
   conda env create -f environment.yml
   ```
3. Install dependencies:
   ```python
   pip install -r requirements.txt
   ```
4. Build project:
   ```python
   python -m build --sdist --wheel .
   ```

### References
- Wikipedia’s article on [Neural Network](https://en.wikipedia.org/wiki/Neural_network).
- The code heavily rely on [PyTorch](https://pytorch.org/docs/stable/index.html).
- To plot the graph’s I referred [Matplotlib](https://matplotlib.org/).
- The code is highly inspired by Morvan Zhou’s work on [simple neural network](https://github.com/MorvanZhou/simple-neural-networks).
- Good Read: [d2l.ai](https://d2l.ai/).
- CSV Dataset is taken from [MNIST datset](https://www.kaggle.com/oddrationale/mnist-in-csv).
