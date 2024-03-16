import numpy as np
import matplotlib.pyplot as plt

import gradient_descent


class LogisticRegression:
    """
    A simple implementation of logistic regression using numpy.

    This class provides functionality for training a logistic regression model
    on labeled binary classification data using gradient descent optimization.
    It supports both batch gradient descent and stochastic gradient descent.

    Attributes
    ----------
    x: np.ndarray
        outputs the features matrix to be used in training the linear regression model.
    y: np.ndarray
        outputs the target vector(matrix) to be used in training the linear regression model.
    epochs: int
        outputs the number of epochs set to run in the implementation of the gradient descent algorithm.
    lr: float
        outputs the learning rate used (to be used) in the algorithm.
    batch: int
        number of datasets to be used for each iteration in training the model. An output of None indicates batch
        gradient descent was used; 1 indicates stochastic gradient descent algorithm was implemented.
    beta: float
        outputs the value for beta (momentum) used.
    weights: np.ndarray
        outputs the weights determined from the implementation of the gradient descent algorithm.
    losses: np.ndarray
        outputs the average loss for each epoch from the implementation of the gradient descent algorithm.

    """

    def __init__(self, x, y, epochs=100, lr=0.1, beta=0, batch=None):
        """
        Initialize regression model for a given input matrix X and target matrix y.
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
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.batch = self.x.shape[0] if batch is None else batch
        self.losses = []
        self.weights = None

    def add_ones(self, x):
        ones_vector = np.ones((self.x.shape[0], 1))
        return np.hstack((ones_vector, x))

    def sigmoid(self, x):
        if x.shape[1] != self.weights.shape[0]:
            x = self.add_ones(x)
        z = x @ self.weights
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        probas = self.sigmoid(x)
        output = (probas >= 0.5).astype(int)
        return output

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

    def fit(self):
        grad_descent = gradient_descent.GradientDescent(self.add_ones(self.x), self.y, self.epochs, self.lr,
                                                        self.beta,self.batch, loss_function=self.cross_entropy,
                                                        grad_function=self.grad_function)
        grad_descent.fit()
        self.weights = grad_descent.weights
        self.losses = grad_descent.losses
