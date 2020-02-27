import numpy as np

class SoftmaxWithLoss:
    """
    The layer calcurate softmax and cross entropy error.

    Attributes
    ----------
    y : vector (or batch)
        The output of softmax function.
    t : vector (or batch)
        The training data. (one hot vector)
    """
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        """
        Parameters
        ----------
        dout : double
            The backpropagation value from next layer.

        Returns
        -------
        dx : vector
            The backpropagation value for previous layer.
        """
        batch_size = self.t.shape[0]
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size

        return dx

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
    
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

