import numpy as np
from .lstm import LSTM

class TimeLSTM:
    def __init__(self, Wx_f, Wh_f, b_f, Wx_i, Wh_i, b_i, Wx_o, Wh_o, b_o, Wx, Wh, b, stateful=False):
        self.params = [Wx_f, Wh_f, b_f, Wx_i, Wh_i, b_i, Wx_o, Wh_o, b_o, Wx, Wh, b]
        self.grads = [np.zeros_like(Wx_f), np.zeros_like(Wh_f), np.zeros_like(b_f), \
                      np.zeros_like(Wx_i), np.zeros_like(Wh_i), np.zeros_like(b_i), \
                      np.zeros_like(Wx_o), np.zeros_like(Wh_o), np.zeros_like(b_o), \
                      np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def set_state(self, h, c=None):
        self.h = h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

    def forward(self, xs):
        Wx_f, Wh_f, b_f, Wx_i, Wh_i, b_i, Wx_o, Wh_o, b_o, Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]


        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx_f, Wh_f, b_f, Wx_i, Wh_i, b_i, Wx_o, Wh_o, b_o, Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0
        grads = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

