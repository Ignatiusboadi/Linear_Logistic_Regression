import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import GradientDescent


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
            number of epochs to be used for the gradient descent algorithm.
        :param lr: float | optional | default = 0.1
            learning rate for the gradient descent algorithm.
        :param beta: float | optional | default = 0.0
            specifies the momentum to be used in the training. Default is 0.0. This implements the gradient descent
            algorithm without momentum.
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
        self.weights = None

    def make_prediction(self, x):
        return x @ self.weights

    def grad_function(self, x, y, weights):
        y = y.reshape((-1, 1))
        grad = 2 * x.T @ (x @ weights - y) / y.shape[0]
        return grad

    def loss_function(self, x, y, weights):
        y = y.reshape((-1, 1))
        return np.mean((y - (x @ weights)) ** 2)

    def fit(self):
        grad_descent = GradientDescent(self.x, self.y, self.epochs, self.lr, self.batch, self.beta,
                                       self.loss_function, self.grad_function)
        grad_descent.sgd_with_momentum()
        self.weights = grad_descent.weights
        self.losses = grad_descent.losses