import torch
from torch import Tensor

class Activation:
    """
    This is the base class of all the activation functions.
    """
    def __init__(self, input: Tensor) -> None:
        super(Activation, self).__init__()
        self.input = input
    
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, *inputs: Tensor) -> Tensor:
        return self.forward(*inputs)

class Linear(Activation):
    """
    Applies linear activation function to the inputs.
    """
    def forward(self, input: Tensor) -> Tensor:
        return input
    
    def backward(self, input: Tensor) -> Tensor:
        return torch.ones_like(input)

class ReLU(Activation):
    """
    Applies the rectified linear unit function element-wise.
    """
    def forward(self, input: Tensor) -> Tensor:
        return torch.max(0, input)

    def backward(self, input: Tensor) -> Tensor:
        return torch.where(input > 0, torch.ones_like(input), torch.ones_like(input))

class LeakyReLU(Activation):
    """
    Applies the leaky relu function element-wise.
    """
    def __init__(self, alpha: float = 0.01) -> None:
        super(LeakyReLU, self).__init__()
        self.alpha = aplha

    def forward(self, input: Tensor) -> Tensor:
        return torch.max(input, self.alpha * input)

    def backward(self, input: Tensor) -> Tensor:
        return torch.where(input > 0., torch.ones_like(input), torch.full_like(input, self.alpha))

def ELU(Activation):
    """
    Applies the exponential linear unit activation function, element-wise.
    """
    def __init__(self, alpha: float = 0.01) -> None:
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        return torch.max(input, self.alpha*(torch.exp(input)-1))

    def backward(self, input: Tensor) -> Tensor:
        return torch.where(input > 0., torch.ones_like(input), self.forward(input) + self.alpha)

class Tanh(Activation):
    """
    Applies tanh activation function element-wise.
    """
    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh(input)

    def backward(self, input: Tensor) -> Tensor:
        return 1. - torch.square(torch.tanh(x))

class Sigmoid(Activation):
    """
    Applies sigmoid function element-wise.
    """
    def forward(self, input: Tensor) -> Tensor:
        return 1./(1. + torch.exp(-input))

    def backward(self, input: Tensor) -> Tensor:
        f = self.forward(input)
        return f*(1.-f)

class SoftPlus(Activation):
    """
    Applies SoftPlus activation element-wise.
    """
    def forward(self, input: Tensor) -> Tensor:
        return torch.log(1. + torch.exp(input))

    def backward(self, input: Tensor) -> Tensor:
        return 1./(1.+torch.exp(-input))

class SoftMax(Activation):
    """
    Applies SoftMax activation function, element-wise.
    """
    def forward(self, x: Tensor, axis: int = -1) -> Tensor:
        shift_x = input - torch.max(input, axis=axis, keepdims=True)
        exp = torch.exp(shift_x + 1e-6)
        return exp/torch.sum(exp, axis=axis, keepdims=True)

    def backward(self, x: Tensor) -> Tensor:
        return torch.ones_like(input)

m = 4111999
relu = ReLU(m)
#leakyrelu = LeakyReLU(m)
elu = ELU(m)
tanh = Tanh(m)
sigmoid = Sigmoid(m)
softplus = SoftPlus(m)
softmax = SoftMax(m)
