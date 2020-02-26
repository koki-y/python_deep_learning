import numpy as np
from embedding import Embedding

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None

    def forward(self, idxs):
        W = self.W
        N, T = idxs.shape
        D = W.shape[1]

        xs = np.empty((N, T, D), dtype='f')
        self.layers = []
        for t in range(T):
            layer = Embedding(*self.params)
            x = layer.forward(idxs[:, t])
            xs[:, t, :] = x
            self.layers.append(layer)

        return xs 

    def backward(self, douts):
        W = self.params
        N, T, D = douts.shape
        V, D = W.shape

        grads = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(douts[:, t, :])

            grads += layer.grads[0]

        self.grads[0][...] = grads
        return None

