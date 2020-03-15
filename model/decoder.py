import numpy as np
from .layer import TimeEmbedding, TimeLSTM, TimeAffine

class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) * 0.01).astype('f')
        lstm_Wx_f = (np.random.randn(D, H) * np.sqrt(D)).astype('f')
        lstm_Wh_f = (np.random.randn(H, H) * np.sqrt(H)).astype('f')
        lstm_b_f  = np.zeros(H).astype('f')
        lstm_Wx_i = (np.random.randn(D, H) * np.sqrt(D)).astype('f')
        lstm_Wh_i = (np.random.randn(H, H) * np.sqrt(H)).astype('f')
        lstm_b_i  = np.zeros(H).astype('f')
        lstm_Wx_o = (np.random.randn(D, H) * np.sqrt(D)).astype('f')
        lstm_Wh_o = (np.random.randn(H, H) * np.sqrt(H)).astype('f')
        lstm_b_o  = np.zeros(H).astype('f')
        lstm_Wx   = (np.random.randn(D, H) * np.sqrt(D)).astype('f')
        lstm_Wh   = (np.random.randn(H, H) * np.sqrt(H)).astype('f')
        lstm_b    = np.zeros(H).astype('f')
        affine_W  = (np.random.randn(H, V) * np.sqrt(H)).astype('f')
        affine_b  = np.zeros(V).astype('f')
        
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx_f, lstm_Wh_f, lstm_b_f, \
                             lstm_Wx_i, lstm_Wh_i, lstm_b_i, \
                             lstm_Wx_o, lstm_Wh_o, lstm_b_o, \
                             lstm_Wx,   lstm_Wh,   lstm_b, \
                             stateful=False)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params = self.embed.params + self.lstm.params + self.affine.params
        self.grads  = self.embed.grads  + self.lstm.grads  + self.affine.params

    def forward(self, xs, h):
        self.lstm.set_state(h)

        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        score = self.affine.forward(hs)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled



