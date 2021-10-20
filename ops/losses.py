import torch
from torch import nn

from activations import softmax

class _Loss:
    """
    The base class to calculate loss.
    """
    def __init__(self, loss, delta) -> None:
        super(Loss, self).__init__()
        self.data = loss
        self.delta = delta

    def __repr__(self) -> str:
        return str(self.data)

class LossFunction:
    def __init__(self) -> None:
        self._pred = None
        self.target = None

    def apply(self, prediction, target):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError

    def _store_pred_target(self, prediction, target):
        p = prediction.data
        p = p if p.dtype is torch.float32 else p.astype(torch.float32)
        self._pred = p
        self._target = target

    def __call__(self, prediction, target):
        return self.apply(prediction, target)

class MSE(LossFunction):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is torch.float32 else target.astype(torch.float32)
        self._store_pred_target(prediction, t)
        loss = torch.mean(torch.square(self._pred - t))/2
        return _Loss(loss, self.delta)

    @property
    def delta(self):
        t = self._target if self._target.dtype is torch.float32 else self._target.astype(torch.float32)
        return self._pred - t

class CrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()
        self._eps = 1e-6

    def apply(self, prediction, target):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError

class SoftMaxCrossEntropy(CrossEntropy):
    def __init__():
        super.__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is torch.float32 else target.astype(torch.float32)
        self._store_pred_target(prediction, t)
        loss = -torch.mean(torch.sum(t * torch.log(self_pred), axis=-1))
        return _Loss(loss, self.data)

    @property
    def delta(self):
        onehot_mask = self._target.astype(torch.bool)
        grad = self._pred.copy()
        grad[onehot_mask] -= 1.
        return grad / len(grad)

class SoftMaxCrossEntropyWithLogits(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is torch.float32 else target.astype(torch.float32)
        self._store_pred_target(prediction, t)
        sm = softmax(self._pred)
        loss = - torch.mean(torch.sum(t * torch.log(sm), axis=-1))
        return _Loss(loss, self.delta)

    @property
    def delta(self):
        grad = softmax(self._pred)
        onehot_mask = self._target.astype(torch.bool)
        grad[onehot_mask] -= 1.
        return grad / len(grad)

class SparseSoftMaxCrossEntropy(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        target = target.astype(torch.int32) if target.dtype is not torch.int32 else target
        self._store_pred_target(prediction, target)
        sm = self._pred
        log_likelihood = torch.log(sm[torch.arange(sm.shape[0]), target.ravel()] + self._eps)
        loss = -torch.mean(log_likelihood)
        return _Loss(loss, self.delta)

    @property
    def delta(self):
        grad = self._pred.copy()
        grad[torch.arange(grad.shape[0]), self._target.ravel()] -= 1.
        return grad / len(grad)

class SparseSoftMaxCrossEntropyWithLogits(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        target = target.astype(torch.int32) if target.dtype is not torch.int32 else target
        self._store_pred_target(prediction, target)
        sm = softmax(self._pred)
        log_likelihood = torch.log(sm[torch.arange(sm.shape[0]), target.ravel()] + self._eps)
        loss = -torch.mean(log_likelihood)
        return _Loss(loss, self.delta)

    @property
    def delta(self):
        grad = softmax(self._pred)
        grad[torch.arange(grad.shape[0]), self._target.ravel()] -= 1.
        return grad / len(grad)

class SigmoidCrossEntropy(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is torch.float32 else target.astype(torch.float32)
        self._store_pred_target(prediction, t)
        p = self._pred
        loss = -torch.mean(
            t * torch.log(p + self._eps) + (1. - t) * torch.log(1 - p + self._eps),
        )
        return _Loss(loss, self.delta)

    @property
    def delta(self):
        t = self._target if self._target.dtype is torch.float32 else self._target.astype(torch.float32)
        return self._pred - t
