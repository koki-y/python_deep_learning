from .attention_weight import AttentionWeight
from .weight_sum       import WeightSum

class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight = AttentionWeight()
        self.weight_sum = WeightSum()
        self.weight = None

    def forward(self, hs, h):
        weight = self.attention_weight.forward(hs, h)
        out = self.weight_sum.forward(hs, weight)
        self.weight = weight
        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum.backward(dout)
        dhs1, dh = self.attention_weight.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

