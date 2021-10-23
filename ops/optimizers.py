import sys
from typing import Tuple

import torch

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/models')
import module as nn

class Optimizer:
    def __init__(self, params: nn.Module, learning_rate: float) -> None:
        self._params = params
        self._learning_rate = learning_rate
        self.vars = []
        self.grads = []
        for layer_p in self._params.values():
            for p_name in layer_p["vars"].keys():
                self.vars.append(layer_p["vars"][p_name])
                self.grads.append(layer_p["grads"][p_name])

    def step(self) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params: nn.Module, learning_rate: float) -> None:
        super().__init__(params=params, learning_rate=learning_rate)
        self.learning_rate = learning_rate

    def step(self) -> None:
        for var, grad in zip(self.vars, self.grads):
            var -= self.learning_rate * grad

class Momentum(Optimizer):
    def __init__(self, params: nn.Module, learning_rate: float = 0.001, momentum: float = 0.9) -> None:
        super().__init__(params=params, learning_rate=learning_rate)
        self._momentum = momentum
        self._mv = [torch.zeros_like(v) for v in self.vars]

    def step(self) -> None:
        for var, grad, mv in zip(self.vars, self.grads, self._mv):
            dv = self._learning_rate * grad
            mv[:] = self._momentum * mv + dv
            var -= mv

class AdaGrad(Optimizer):
    def __init__(self, params: nn.Module, learning_rate: float = 0.01, eps: float = 1e-06) -> None:
        super().__init__(params=params, learning_rate=learning_rate)
        self._eps = eps
        self._v = [torch.zeros_like(v) for v in self.vars]

    def step(self) -> None:
        for var, grad, v in zip(self.vars, self.grads, self._v):
            v += torch.square(grad)
            var -= self._learning_rate * grad / torch.sqrt(v + self._eps)

class Adadelta(Optimizer):
    def __init__(self, params: nn.Module, learning_rate: float = 1., rho: float = 0.9, eps: float = 1e-06) -> None:
        super().__init__(params=params, learning_rate=learning_rate)
        self._rho = rho
        self._eps = eps
        self._m = [torch.zeros_like(v) for v in self.vars]
        self._v = [torch.zeros_like(v) for v in self.vars]

    def step(self) -> None:
        for var, grad, m, v in zip(self.vars, self.grads, self._m, self._v):
            v[:] = self._rho * v + (1. - self._rho) * torch.square(grad)
            delta = torch.sqrt(m + self._eps) / torch.sqrt(v + self._eps) * grad
            var -= self._learning_rate * delta
            m[:] = self._rho * m + (1. - self._rho) * torch.square(delta)

class RMSProp(Optimizer):
    def __init__(self, params: nn.Module, learning_rate: float = 0.01, alpha: float = 0.99, eps: float = 1e-08) -> None:
        super().__init__(params=params, learning_rate=learning_rate)
        self._alpha = alpha
        self._eps = eps
        self._v = [torch.zeros_like(v) for v in self.vars]

    def step(self) -> None:
        for var, grad, v in zip(self.vars, self.grads, self._v):
            v[:] = self._alpha * v + (1. - self._alpha) * torch.square(grad)
            var -= self._learning_rate * grad / torch.sqrt(v + self._eps)

class Adam(Optimizer):
    def __init__(self, params: nn.Module, learning_rate: None = 0.01, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08) -> None:
        super().__init__(params=params, learning_rate=learning_rate)
        self._betas = betas
        self._eps = eps
        self._m = [torch.zeros_like(v) for v in self.vars]
        self._v = [torch.zeros_like(v) for v in self.vars]

    def step(self) -> None:
        b1, b2 = self._betas
        b1_crt, b2_crt = b1, b2
        for var, grad, m, v in zip(self.vars, self.grads, self._m, self._v):
            m[:] = b1 * m + (1. - b1) * grad
            v[:] = b2 * v + (1. - b2) * torch.square(grad)
            b1_crt, b2_crt = b1_crt * b1, b2_crt * b2   # bias correction
            m_crt = m / (1. - b1_crt)
            v_crt = v / (1. - b2_crt)
            var -= self._learning_rate * m_crt / torch.sqrt(v_crt + self._eps)

class AdaMax(Optimizer):
    def __init__(self, params: nn.Module, learning_rate: float = 0.01, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08) -> None:
        super().__init__(params=params, learning_rate=learning_rate)
        self._betas = betas
        self._eps = eps
        self._m = [torch.zeros_like(v) for v in self.vars]
        self._v = [torch.zeros_like(v) for v in self.vars]

    def step(self) -> None:
        b1, b2 = self._betas
        b1_crt = b1
        for var, grad, m, v in zip(self.vars, self.grads, self._m, self._v):
            m[:] = b1 * m + (1. - b1) * grad
            v[:] = torch.max(b2 * v, torch.abs(grad))
            b1_crt = b1_crt * b1  # bias correction
            m_crt = m / (1. - b1_crt)
            var -= self._lr * m_crt / (v + self._eps)
