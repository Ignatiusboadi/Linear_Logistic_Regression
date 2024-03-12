import numpy as np


class GradientDescent:
    """
    This class implements the gradient descent algorithm. Depending on the values passed, batch, mini-batch or
    stochastic gradient descent is implemented. If beta is 0, the algorithms are implemented without momentum, otherwise
    it is implemented with momentum.

    Attributes
    ----------
    epochs: int
        outputs the number of epochs set to run in the implementation of the gradient descent algorithm.
    lr: float
        outputs the learning rate used (to be used) in the algorithm.
    batch: int
        number of datasets to be used for each iteration in training the model. An output of None indicates batch
        gradient descent was used; 1 indicates stochastic gradient descent algorithm was implemented.
    beta: float
        outputs the value for beta (momentum) used.
    loss_function: function
        outputs function used as loss function in the implementation of the algorithm.
    grad_function: function
        outputs the gradient of the loss function.

    """

    def __init__(self, x, y, epochs, lr, batch, beta, loss_function, grad_function):
        self.x = x
        self.y = y
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.beta = beta
        self.loss_function = loss_function
        self.grad_function = grad_function
        self.weights = None
