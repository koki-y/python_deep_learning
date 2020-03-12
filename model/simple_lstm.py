import numpy as np
from .layer import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss

class SimpleLstm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W   = (np.random.randn(V, D) * 0.01).astype('f')
        lstm_Wx_f = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wh_f = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_f  = np.zeros(H).astype('f')
        lstm_Wx_i = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wh_i = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_i  = np.zeros(H).astype('f')
        lstm_Wx_o = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wh_o = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_o  = np.zeros(H).astype('f')
        lstm_Wx_g = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wh_g = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_g  = np.zeros(H).astype('f')
        affine_W  = (np.random.randn(H, V) / np.sqrt(H)).astype('f')
        affine_b  = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx_f, lstm_Wh_f, lstm_b_f, \
                     lstm_Wx_i, lstm_Wh_i, lstm_b_i, \
                     lstm_Wx_o, lstm_Wh_o, lstm_b_o, \
                     lstm_Wx_g, lstm_Wh_g, lstm_b_g, \
                     stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer  = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads  += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        xs = self.predict(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()

    def set_params(self, params):
        for i, param in enumerate(self.params):
            param[...] = params[i]

