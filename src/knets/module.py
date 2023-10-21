import sys
from typing import List, Optional

import torch
from torch import Tensor

# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import losses
import layers

class Module(object):
    def __init__(
            self
    ) -> None:
        self._ordered_layers = []
        self.params = {}

    def forward(
            self,
            *inputs: List[Tensor]
    ) -> Optional[Tensor]:
        raise NotImplementedError

    def backward(
            self,
            loss: losses._Loss
    ) -> None:
        assert isinstance(loss, losses._Loss)
        # find net order
        _layers = []
        for name, v in self.__dict__.items():
            if not isinstance(v, layers._BaseLayer):
                continue
            layer = v
            layer.name = name
            _layers.append((layer.order, layer))
        self._ordered_layers = [l[1] for l in sorted(_layers, key=lambda x: x[0])]

        # back propagate through this order
        last_layer = self._ordered_layers[-1]
        last_layer.data_vars["out"].set_error(loss.delta)
        for layer in self._ordered_layers[::-1]:
            grads = layer.backward()
            if isinstance(layer, layers.ParamLayer):
                for k in layer.param_vars.keys():
                    self.params[layer.name]["grads"][k][:] = grads[k]

    def sequential(
            self,
            *layers
    ):
        assert isinstance(layers, (list, tuple))
        for i, l in enumerate(layers):
            self.__setattr__("layer_%i" % i, l)
        return SeqLayers(layers)

    def __call__(
            self,
            *args
    ):
        return self.forward(*args)

    def __setattr__(
            self,
            key,
            value
    ):
        if isinstance(value, layers.ParamLayer):
            layer = value
            self.params[key] = {
                "vars": layer.param_vars,
                "grads": {k: torch.empty_like(layer.param_vars[k]) for k in layer.param_vars.keys()}
            }
        object.__setattr__(self, key, value)


class SeqLayers:
    def __init__(
            self,
            layers
    ):
        assert isinstance(layers, (list, tuple))
        for l in layers:
            assert isinstance(l, layers.BaseLayer)
        self.layers = layers

    def forward(
            self,
            x
    ):
        for l in self.layers:
            x = l.forward(x)
        return x

    def __call__(
            self,
            x
    ):
        return self.forward(x)
