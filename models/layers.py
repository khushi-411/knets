import sys
from typing import Dict

import torch
from torch import Tensor

sys.path.insert(1, '/home/khushi/Documents/simple-neural-network/ops')
import variable
import activations as act
import initializers as init

class _BaseLayer:
    def __init__(
            self
    ) -> None:
        self.order = None
        self.name = None
        self._x = None
        self.data_vars = {}

    def forward(
            self,
            x: Tensor
    ):
        raise NotImplementedError

    def backward(
            self
    ):
        raise NotImplementedError

    def _process_input(
            self,
            x: Tensor
    ) -> Tensor:
        if isinstance(x, Tensor):
            # https://discuss.pytorch.org/t/how-to-cast-a-tensor-to-another-type/2713
            x = x.to(torch.float32)
            x = variable.Variable(x)
            x.info["new_layer_order"] = 0

        self.data_vars["in"] = x
        # x is Variable, extract _x value from x.data
        self.order = x.info["new_layer_order"]
        _x = x.data
        return _x

    def _wrap_out(
            self,
            out: Tensor
    ) -> variable.Variable:
        out = variable.Variable(out)
        out.info["new_layer_order"] = self.order + 1
        self.data_vars["out"] = out     # add to layer's data_vars
        return out

    def __call__(
            self,
            x: Tensor
    ) -> variable.Variable:
        return self.forward(x)

class ParamLayer(_BaseLayer):
    def __init__(
            self,
            w_shape,
            activation,
            w_initializer,
            b_initializer,
            use_bias
    ) -> None:
        super().__init__()
        self.param_vars = {}
        self.w = torch.empty(w_shape, dtype=torch.float32)
        self.param_vars["w"] = self.w
        if use_bias:
            shape = [1]*len(w_shape)
            shape[-1] = w_shape[-1]     # only have bias on the last dimension
            self.b = torch.empty(shape, dtype=torch.float32)
            self.param_vars["b"] = self.b
        self.use_bias = use_bias

        if activation is None:
            # TODO
            self._a = act.Linear()
        elif isinstance(activation, act.Activation):
            self._a = activation
        else:
            raise TypeError

        if w_initializer is None:
            init.TruncatedNormal(0., 0.01).initialize(self.w)
        elif isinstance(w_initializer, init._BaseInitializer):
            w_initializer.initialize(self.w)
        else:
            raise TypeError

        if use_bias:
            if b_initializer is None:
                init.Constant(0.01).initialize(self.b)
            elif isinstance(b_initializer, init._BaseInitializer):
                b_initializer.initialize(self.b)
            else:
                raise TypeError

        self._wx_b = None
        self._activated = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Dense(ParamLayer):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            activation=None,  # activations.Tanh
            w_initializer=None, # initializers.RandomUniform
            b_initializer=None,  # initializers.Constant
            use_bias: bool = True,
    ) -> None:
        super().__init__(
            w_shape=(n_in, n_out),
            activation=activation,
            w_initializer=w_initializer,
            b_initializer=b_initializer,
            use_bias=use_bias
        )

        self._n_in = n_in
        self._n_out = n_out

    def forward(
            self,
            x: Tensor
    ) -> variable.Variable:
        self._x = self._process_input(x)
        # https://stackoverflow.com/questions/66720543/pytorch-1d-tensors-expected-but-got-2d-tensors
        self._wx_b = self._x.matmul(self.w)
        if self.use_bias:
            self._wx_b += self.b

        self._activated = self._a(self._wx_b)   # if act is None, act will be Linear
        wrapped_out = self._wrap_out(self._activated)
        return wrapped_out

    def backward(
            self
    ) -> Dict[Tensor, Tensor]:
        # dw, db
        dz = self.data_vars["out"].error
        dz *= self._a.derivative(self._wx_b)
        grads = {"w": self._x.T.matmul(dz)}
        if self.use_bias:
            grads["b"] = torch.sum(dz, axis=0, keepdims=True)
        # dx
        self.data_vars["in"].set_error(dz.matmul(self.w.T))     # pass error to the layer before
        return grads

class Conv2D(ParamLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            channels_last=True,
            activation=None,
            w_initializer=None,
            b_initializer=None,
            use_bias=True,
    ):
        self.kernel_size = get_tuple(kernel_size)
        self.strides = get_tuple(strides)
        super().__init__(
                w_shape=(in_channels,) + self.kernel_size + (out_channels,),
                activation=activation,
                w_initializer=w_initializer,
                b_initializer=b_initializer,
                use_bias=use_bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # https://stackoverflow.com/questions/53986123
        self.padding = padding.lower()
        assert padding in ("valid", "same"), ValueError

        self.channels_last = channels_last
        self._padded = None
        self._p_tblr = None     # padded dim from top, bottom, left, right

    def forward(
            self,
            x
    ):
        self._x = self._process_input(x)
        if not self.channels_last:
            # [batch, channel, height, width] => [batch, height, width, channel]
            self._x = torch.transpose(self._x, (0, 2, 3, 1))
        self._padded, tmp_conved, self._p_tblr = get_padded_and_tmp_out(
            self._x, self.kernel_size, self.strides, self.out_channels, self.padding)

        # convolution
        self._wx_b = self.convolution(self._padded, self.w, tmp_conved)
        if self.use_bias:   # tied biases
            self._wx_b += self.b

        self._activated = self._a(self._wx_b)
        wrapped_out = self._wrap_out(
            self._activated if self.channels_last else self._activated.transpose((0, 3, 1, 2)))
        
        return wrapped_out

    def backward(
            self
    ):
        dz = self.data_vars["out"].error
        dz *= self._a.derivative(self._wx_b)

        # dw, db
        dw = torch.empty_like(self.w)  # [c, h, w, out]
        dw = self.convolution(self._padded.transpose((3, 1, 2, 0)), dz, dw)

        grads = {"w": dw}
        if self.use_bias:   # tied biases
            grads["b"] = np.sum(dz, axis=(0, 1, 2), keepdims=True)

        # dx
        padded_dx = torch.zeros_like(self._padded)    # [n, h, w, c]
        s0, s1, k0, k1 = self.strides + self.kernel_size
        t_flt = self.w.transpose((3, 1, 2, 0))  # [c, fh, hw, out] => [out, fh, fw, c]
        
        for i in range(dz.shape[1]):
            for j in range(dz.shape[2]):
                padded_dx[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :] += dz[:, i, j, :].reshape((-1, self.out_channels)).dot(
                                                            t_flt.reshape((self.out_channels, -1))
                                                        ).reshape((-1, k0, k1, padded_dx.shape[-1]))
        t, b, l, r = [self._p_tblr[0], padded_dx.shape[1] - self._p_tblr[1],
                      self._p_tblr[2], padded_dx.shape[2] - self._p_tblr[3]]
        self.data_vars["in"].set_error(padded_dx[:, t:b, l:r, :])      # pass error to the layer before
        
        return grads

    def convolution(
            self,
            x,
            flt,
            conved
    ):
        
        batch_size = x.shape[0]
        t_flt = flt.transpose((1, 2, 0, 3))  # [c,h,w,out] => [h,w,c,out]
        s0, s1, k0, k1 = self.strides + tuple(flt.shape[1:3])
        
        for i in range(0, conved.shape[1]):  # in each row of the convoluted feature map
            for j in range(0, conved.shape[2]):  # in each column of the convoluted feature map
                x_seg_matrix = x[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :].reshape((batch_size, -1))  # [n,h,w,c] => [n, h*w*c]
                flt_matrix = t_flt.reshape((-1, flt.shape[-1]))  # [h,w,c, out] => [h*w*c, out]
                filtered = x_seg_matrix.dot(flt_matrix)  # sum of filtered window [n, out]
                conved[:, i, j, :] = filtered
        
        return conved

    def fast_convolution(
            self,
            x,
            flt,
            conved
    ):
        # create patch matrix
        oh, ow, sh, sw, fh, fw = [conved.shape[1], conved.shape[2], self.strides[0],
                                  self.strides[1], flt.shape[1], flt.shape[2]]
        n, h, w, c = x.shape
        shape = (n, oh, ow, fh, fw, c)
        strides = (c * h * w, sh * w, sw, w, 1, h * w)
        strides = x.itemsize * np.array(strides)
        x_col = torch.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)
        x_col = torch.ascontiguousarray(x_col)
        x_col.shape = (n * oh * ow, fh * fw * c)    # [n*oh*ow, fh*fw*c]
        self._padded_col = x_col       # padded [n,h,w,c] => [n*oh*ow, h*w*c]
        w_t = flt.transpose((1, 2, 0, 3)).reshape(-1, self.out_channels)  # => [hwc, oc]

        # IMPORTANT! as_stride function has some wired behaviours
        # which gives a not accurate result (precision issue) when performing matrix dot product.
        # I have compared the fast convolution with normal convolution and cannot explain the precision issue.
        wx = self._padded_col.dot(w_t)  # [n*oh*ow, fh*fw*c] dot [fh*fw*c, oc] => [n*oh*ow, oc]
        
        return wx.reshape(conved.shape)

    def fast_backward(
            self
    ):
        
        dz = self.data_vars["out"].error
        dz *= self._a.derivative(self._wx_b)

        # dw, db
        dz_reshape = dz.reshape(-1, self.out_channels)      # => [n*oh*ow, oc]
        # self._padded_col.T~[fh*fw*c, n*oh*ow] dot [n*oh*ow, oc] => [fh*fw*c, oc]
        dw = self._padded_col.T.dot(dz_reshape).reshape(self.kernel_size[0], self.kernel_size[1], -1, self.out_channels)
        dw = dw.transpose(2, 0, 1, 3)   # => [c, fh, fw, oc]
        grads = {"w": dw}

        if self.use_bias:  # tied biases
            grads["b"] = torch.sum(dz, axis=(0, 1, 2), keepdims=True)

        # dx
        padded_dx = torch.zeros_like(self._padded)  # [n, h, w, c]
        s0, s1, k0, k1 = self.strides + self.kernel_size
        t_flt = self.w.transpose((3, 1, 2, 0))  # [c, fh, hw, out] => [out, fh, fw, c]
        
        for i in range(dz.shape[1]):
            for j in range(dz.shape[2]):
                padded_dx[:, i * s0:i * s0 + k0, j * s1:j * s1 + k1, :] += dz[:, i, j, :].reshape(
                    (-1, self.out_channels)).dot(
                    t_flt.reshape((self.out_channels, -1))
                ).reshape((-1, k0, k1, padded_dx.shape[-1]))
        
        t, b, l, r = self._p_tblr[0], padded_dx.shape[1] - self._p_tblr[1], self._p_tblr[2], padded_dx.shape[2] - \
                     self._p_tblr[3]
        self.data_vars["in"].set_error(padded_dx[:, t:b, l:r, :])      # pass the error to the layer before
        
        return grads

class Pool_(_BaseLayer):

    def __init__(
            self,
            kernal_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            channels_last=True,
    ):
        super().__init__()
        self.kernel_size = get_tuple(kernal_size)
        self.strides = get_tuple(strides)
        self.padding = padding.lower()
        assert padding in ("valid", "same"), ValueError
        self.channels_last = channels_last
        self._padded = None
        self._p_tblr = None

    def forward(
            self,
            x
    ):
        self._x = self._process_input(x)
        
        if not self.channels_last:
            # [batch, channel, height, width] => [batch, height, width, channel]
            self._x = torch.transpose(self._x, (0, 2, 3, 1))
        self._padded, out, self._p_tblr = get_padded_and_tmp_out(
            self._x, self.kernel_size, self.strides, self._x.shape[-1], self.padding)
        s0, s1, k0, k1 = self.strides + self.kernel_size
        
        for i in range(0, out.shape[1]):  # in each row of the convoluted feature map
            for j in range(0, out.shape[2]):  # in each column of the convoluted feature map
                window = self._padded[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :]  # [n,h,w,c]
                out[:, i, j, :] = self.agg_func(window)
        
        wrapped_out = self._wrap_out(out if self.channels_last else out.transpose((0, 3, 1, 2)))
        
        return wrapped_out

    def backward(self):
        raise NotImplementedError

    @staticmethod
    def agg_func(x):
        raise NotImplementedError

class MaxPool2D(Pool_):
    
    def __init__(
            self,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            channels_last=True,
    ):
        
        super().__init__(
            kernal_size=pool_size,
            strides=strides,
            padding=padding,
            channels_last=channels_last,)

    @staticmethod
    def agg_func(x):
        return x.max(axis=(1, 2))

    def backward(
            self
    ):
        dz = self.data_vars["out"].error
        grad = None
        s0, s1, k0, k1 = self.strides + self.kernel_size
        padded_dx = torch.zeros_like(self._padded)  # [n, h, w, c]
        
        for i in range(dz.shape[1]):
            for j in range(dz.shape[2]):
                window = self._padded[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :]  # [n,fh,fw,c]
                window_mask = window == torch.max(window, axis=(1, 2), keepdims=True)
                window_dz = dz[:, i:i+1, j:j+1, :] * window_mask.astype(torch.float32)
                padded_dx[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :] += window_dz
        
        t, b, l, r = [self._p_tblr[0], padded_dx.shape[1]-self._p_tblr[1],
                      self._p_tblr[2], padded_dx.shape[2]-self._p_tblr[3]]
        self.data_vars["in"].set_error(padded_dx[:, t:b, l:r, :])      # pass the error to the layer before
        return grad

class AvgPool2D(Pool_):
    
    def __init__(
            self,
            kernal_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            channels_last=True,
    ):
        super().__init__(
            kernal_size=kernal_size,
            strides=strides,
            padding=padding,
            channels_last=channels_last,)

    @staticmethod
    def agg_func(x):
        return x.mean(axis=(1, 2))

    def backward(self):
        
        dz = self.data_vars["out"].error
        grad = None
        s0, s1, k0, k1 = self.strides + self.kernel_size
        padded_dx = torch.zeros_like(self._padded)  # [n, h, w, c]
        
        for i in range(dz.shape[1]):
            for j in range(dz.shape[2]):
                window_dz = dz[:, i:i + 1, j:j + 1, :] * np.full(
                    (1, k0, k1, dz.shape[-1]), 1./(k0*k1), dtype=torch.float32)
                padded_dx[:, i * s0:i * s0 + k0, j * s1:j * s1 + k1, :] += window_dz
        
        t, b, l, r = [self._p_tblr[0], padded_dx.shape[1] - self._p_tblr[1],
                      self._p_tblr[2], padded_dx.shape[2] - self._p_tblr[3]]
        
        self.data_vars["in"].set_error(padded_dx[:, t:b, l:r, :])  # pass the error to the layer before

        return grad

class Flatten(_BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._x = self._process_input(x)
        out = self._x.reshape((self._x.shape[0], -1))
        wrapped_out = self._wrap_out(out)
        return wrapped_out

    def backward(self):
        dz = self.data_vars["out"].error
        grad = None
        self.data_vars["in"].set_error(dz.reshape(self._x.shape))
        return grad

def get_tuple(
        inputs
):
    if isinstance(inputs, (tuple, list)):
        out = tuple(inputs)
    elif isinstance(inputs, int):
        out = (inputs, inputs)
    else:
        raise TypeError
    return out

def get_padded_and_tmp_out(
        img,
        kernel_size,
        strides,
        out_channels,
        padding
):

    batch, h, w = img.shape[:3]
    (fh, fw), (sh, sw) = kernel_size, strides

    if padding == "same":
        out_h = int(torch.ceil(h / sh))
        out_w = int(torch.ceil(w / sw))
        ph = int(torch.max([0, (out_h - 1) * sh + fh - h]))
        pw = int(torch.max([0, (out_w - 1) * sw + fw - w]))
        pt, pl = int(torch.floor(ph / 2)), int(torch.floor(pw / 2))
        pb, pr = ph - pt, pw - pl
    
    elif padding == "valid":
        out_h = int(torch.ceil((h - fh + 1) / sh))
        out_w = int(torch.ceil((w - fw + 1) / sw))
        pt, pb, pl, pr = 0, 0, 0, 0
    
    else:
        raise ValueError
    
    padded_img = torch.pad(img, ((0, 0), (pt, pb), (pl, pr), (0, 0)), 'constant', constant_values=0.).astype(torch.float32)
    tmp_out = torch.zeros((batch, out_h, out_w, out_channels), dtype=torch.float32)
    
    return padded_img, tmp_out, (pt, pb, pl, pr)
