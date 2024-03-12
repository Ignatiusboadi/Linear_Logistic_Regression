import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """

    """
    def __init__(self, x, y, epochs=100, lr=0.1, beta=0, batch=None):
        """

        :param x: np.ndarray
            input matrix
        :param y: np.ndarray
            target vector (matrix)
        :param epochs: int
            number of epochs
        :param lr: float
            learning rate
        :param beta: float
            momentum
        :param batch: int

        """
        self.x = x
        self.y = y
        self.n_epochs = epochs
        self.lr = lr
        self.beta = beta
        self.batch = self.x.shape[0] if batch is None else batch
        self.train_losses = []
        self.w = None
        self.weight = []

    def cross_entropy(self, x, y, w):
        y = y.reshape(-1, 1)
        z = x @ w
        y_pred = 1 / (1 + np.exp(-z))
        loss = -1 * np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def grad_function(self, x, y, w):
        y = y.reshape(-1, 1)
        z = x @ w
        y_pred = 1 / (1 + np.exp(-z))
        return -1 / x.shape[0] * x.T @ (y - y_pred)