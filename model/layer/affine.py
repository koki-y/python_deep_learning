import numpy as np

class Affine:
    """
    Affine calculation layer.

    Attributes
    ---------------
    params : list
        The model parameters. (weight, bias)
    grads : list
        The learning gradients. (weight, bias)
    x : list
        The input of affine calculation.
    """

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        """
        Return the result of affine calculation.

        Parameters
        ----------
        x : vector (or matrix)
            The input of affine calculation.

        Returns
        -------
        out : vector (or matrix)
            The output of affine calculation.
        """
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out
