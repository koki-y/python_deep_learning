from .attention_encoder import AttentionEncoder
from .attention_decoder import AttentionDecoder
from .layer             import TimeSoftmaxWithLoss

class AttentionSeq2Seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(V, D, H)
        self.decoder = AttentionDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads  = self.encoder.grads  + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        hs    = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, hs)
        loss  = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
