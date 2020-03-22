import numpy as np
from .layer import TimeEmbedding, TimeLSTM

class AttentionEncoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) * 0.01).astype('f')
        lstm_Wx_f = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wh_f = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_f  = np.zeros(H).astype('f')
        lstm_Wx_i = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wh_i = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_i  = np.zeros(H).astype('f')
        lstm_Wx_o = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wh_o = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b_o  = np.zeros(H).astype('f')
        lstm_Wx   = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wh   = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        lstm_b    = np.zeros(H).astype('f')
        
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx_f, lstm_Wh_f, lstm_b_f, \
                             lstm_Wx_i, lstm_Wh_i, lstm_b_i, \
                             lstm_Wx_o, lstm_Wh_o, lstm_b_o, \
                             lstm_Wx,   lstm_Wh,   lstm_b, \
                             stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads  = self.embed.grads  + self.lstm.grads

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
