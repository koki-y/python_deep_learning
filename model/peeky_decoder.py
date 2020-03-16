import numpy as np
from .layer import TimeEmbedding, TimeLSTM, TimeAffine

class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) * 0.01).astype('f')
        lstm_Wx_f = (np.random.randn(H + D, H) / np.sqrt(H + D)).astype('f')
        lstm_Wh_f = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_f  = np.zeros(H).astype('f')
        lstm_Wx_i = (np.random.randn(H + D, H) / np.sqrt(H + D)).astype('f')
        lstm_Wh_i = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_i  = np.zeros(H).astype('f')
        lstm_Wx_o = (np.random.randn(H + D, H) / np.sqrt(H + D)).astype('f')
        lstm_Wh_o = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_o  = np.zeros(H).astype('f')
        lstm_Wx   = (np.random.randn(H + D, H) / np.sqrt(H + D)).astype('f')
        lstm_Wh   = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b    = np.zeros(H).astype('f')
        affine_W  = (np.random.randn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b  = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx_f, lstm_Wh_f, lstm_b_f, \
                             lstm_Wx_i, lstm_Wh_i, lstm_b_i, \
                             lstm_Wx_o, lstm_Wh_o, lstm_b_o, \
                             lstm_Wx,   lstm_Wh,   lstm_b, \
                             stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params = self.embed.params + self.lstm.params + self.affine.params
        self.grads  = self.embed.grads  + self.lstm.grads  + self.affine.grads

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score

    def backward(self, dscore):
        H = self.cache
        dout = self.affine.backward(dscore)

        dout, dh1 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)

        dout, dh2 = dout[:, :, H:], dout[:, :, :H]
        dout = self.embed.backward(dout)

        dh = self.lstm.dh + np.sum((dh1 + dh2), axis=1)
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)

            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled

