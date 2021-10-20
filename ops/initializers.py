from typing import List, Union

import torch
from torch import Tensor, nn

class _BaseInitializer:
    def initialize(
            self, 
            x: Union[List[Tensor], Tensor]
    ):
        raise NotImplementedError

class RandomNormal(_BaseInitializer):
    def __init__(
            self,
            mean: float = 0.,
            std: float = 1.
    ) -> None:
        self._mean = mean
        self._std = std

    def initialize(
            self,
            x: Union[List[Tensor], Tensor]
    ) -> None:
        x[:] = torch.nn.init.normal_(x, mean=self._mean, std=self._std)

class RandomUniform(_BaseInitializer):
    def __init__(
            self,
            low: float = 0.,
            high: float = 1.
    ) -> None:
        self._low = low
        self._high = high

    def initialize(
            self,
            x: Union[List[Tensor], Tensor]
    ) -> None:
        # https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        x[:] = torch.nn.init.uniform_(x, self._low, self._high)

class Zeros(_BaseInitializer):
    def initialize(
            self,
            x: Union[List[Tensor], Tensor]
    ) -> None:
        x[:] = torch.zeros_like(x)

class Ones(_BaseInitializer):
    def initialize(
            self,
            x: Union[List[Tensor], Tensor]
    ) -> None:
        x[:] = torch.ones_like(x)

class TruncatedNormal(_BaseInitializer):
    def __init__(
            self,
            mean: float = 0.,
            std: float = 1.
    ) -> None:
        self._mean = mean
        self._std = std

    def initialize(
            self,
            x: Union[List[Tensor], Tensor]
    ) -> None:
        x[:] = torch.nn.init.normal_(x, mean=self._mean, std=self._std)
        truncated = 2*self._std + self._mean
        x[:] = torch.clip(x, -truncated, truncated)

class Constant(_BaseInitializer):
    def __init__(
            self,
            v: Union[List[Tensor], Tensor]
    ) -> None:
        self._v = v

    def initialize(
            self,
            x: Union[List[Tensor], Tensor]
    ) -> None:
        x[:] = torch.full_like(x, self._v)

random_normal = RandomNormal()
random_uniform = RandomUniform()
zeros = Zeros()
ones = Ones()
truncated_normal = TruncatedNormal()
