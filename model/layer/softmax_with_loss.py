import numpy as np

class SoftmaxWithLoss:
    """
    The layer calcurate softmax and cross entropy error.

    Attributes
    ----------
    loss : real number (or batch)
    y : vector (or batch)
        The output of softmax function.
    t : vector (or batch)
        The training data. (one hot vector)
    """
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = SoftmaxWithLoss.softmax(x)
        self.loss = SoftmaxWithLoss.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """
        Parameters
        ----------
        d_out : double
            The backpropagation value from next layer.

        Returns
        -------
        out : vector
            The backpropagation value for previous layer.
        """
        batch_size = self.t.shape[0]
        out = (self.y - self.t) / batch_size

        return out

    def softmax(x):
        if x.ndim == 2:
            x = x.T # prevent overflow
            x = x -np.max(x, axis=0)
            out = np.exp(x) / np.sum(np.exp(x), axis=0)
            return out.T
        else:
            x = x - np.max(x) # prevent overflow
            out = np.exp(x) / np.sum(np.exp(x))
            return out

    def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        delta = 1e-7
        return -np.sum(t * np.log(y + delta)) / batch_size

