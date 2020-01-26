import numpy as np

class Sigmoid:
    """
    Sigmoid function layer.

    Attributes
    ---------------
    params : list
        The model parameters. (weight, bias)
        There's no elements for this layer.
    grads : list
        The learning gradients. (weight, bias)
        There's no elements for this layer.
    out : double
        The output of forwarding.
    """

    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        """
        Return the result of sigmoid function.
        If the input is vector, it returns vector as result.

        Parameters
        ----------
        x : Real number (or vector)
            The input of sigmoid function.

        Returns
        -------
        out : Real number (or vector)
            The output of sigmoid function.
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, d_out):
        """
        Parameters
        ----------
        d_out : double
            The backpropagation value from next layer.

        Returns
        -------
        out : double
            The backpropagation value for previous layer.
        """
        return d_out * (1.0 - self.out) * self.out

