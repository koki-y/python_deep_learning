import numpy as np

class SoftmaxWithLoss:
    def softmax(x):
        k = np.max(x)         # prevent overflow
        exp_x = np.exp(x - k) # 
        sum_exp = np.sum(exp_x)
        out = exp_x / sum_exp

        return out

    def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        delta = 1e-7
        return -np.sum(t * np.log(y + delta)) / batch_size
