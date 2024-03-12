import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    This class implements linear regression for given input matrix and output vector (matrix).
    Attributes
    -----------


    Methods
    --------

    """

    def __init__(self, x, y, epochs=1000, lr=0.1, batch=None, beta=0.0):
        """
        Initialize regression model for given input matrix X and target matrix y.
        :param X: np.ndarray
            Input matrix X
        :param y: np.ndarray
            target matrix y
        :param epochs: int | optional | default = 1000
            number of epochs
        :param lr: float | optional | default = 0.1
            learning rate
        :param beta: float | optional | default = 0.0
            momentum
        :param batch: int
            if None, batch gradient descent will be used for the Gradient descent algorithm, 1 for stochastic gradient
            descent and an integer for the number of batches in each batch for minibatch gradient descent.
        """
        self.x = x
        self.y = y
        self.epochs = epochs
        self.beta = beta
        self.lr = lr
        self.batch = batch
