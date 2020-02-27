import numpy as np
from .softmax_with_loss import SoftmaxWithLoss

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.layers = None

    def forward(self, xs, ts):
        N, T, V = xs.shape

        self.layers = []
        total_loss = 0
        for t in range(T):
            layer = SoftmaxWithLoss()
            total_loss += layer.forward(xs[:, t, :], ts[:, t])
            self.layers.append(layer)

        self.cache = (N, T, V)

        return total_loss / T

    def backward(self, dout=1):
        layers = self.layers
        N, T, V = self.cache
        douts = np.empty((N, T, V), dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = layers[t]
            douts[:, t, :] = layer.backward(dout) # * T

        return douts

