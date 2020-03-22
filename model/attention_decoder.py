import numpy as np
from .layer import TimeEmbedding, TimeLSTM, TimeAffine, TimeAttention

class AttentionDecoder:
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
        affine_W  = (np.random.randn(2 * H, V) / np.sqrt(2 * H)).astype('f')
        affine_b  = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx_f, lstm_Wh_f, lstm_b_f, \
                             lstm_Wx_i, lstm_Wh_i, lstm_b_i, \
                             lstm_Wx_o, lstm_Wh_o, lstm_b_o, \
                             lstm_Wx,   lstm_Wh,   lstm_b, \
                             stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)

        self.layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads  += layer.grads

    def forward(self, xs, enc_hs):
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled

