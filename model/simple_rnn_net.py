import numpy as np

class SimpleRnnNet:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W  = (np.random.randn(V, D) * 0.01).astype('f')
        rnn_Wx   = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh   = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        rnn_b    = np.zeros(H).astype('f')
        affine_W = (np.random.randn(H, V) / rp.sqrt(H)).astype('f')
        affine_b = np.zeros(H).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer  = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads  += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()

