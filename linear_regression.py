import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import GradientDescent


class LinearRegression:
    """
    This class provides functionality for training a linear regression model on a features matrix
    and target vector using gradient descent optimization. It supports both batch gradient descent
    and stochastic gradient descent. The user can specify  the number of epochs, the learning rate
    for the algorithm and beta, if they want to apply momentum.

    Attributes
    -----------
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

    Methods
    --------
    fit:
        implements the gradient descent algorithm using the given features matrix and target vector to find
        the best weights.
    make_prediction:
        makes prediction using a given features matrix and the weights determined from the Gradient descent
        algorithm.
    loss_function:
        computes the loss using given weights, a feature matrix and a target vector.
    grad_function:
        computes the gradient of the loss using given weights, a feature matrix and a target vector.
    plot_function:
        plots the graphs of the target vector against one column feature matrix, and the predicted y value against
        the one column feature matrix.
    r_square:
        computes the r-square on the training set if no data is passed. When a dataset is passed,
        makes a prediction using x and the weights, then computes the r-square.
    """

    def __init__(self, x, y, epochs=1000, lr=0.1, batch=None, beta=0.0):
        """
        Initialize regression model for a given input matrix X and target matrix y.
        :param x: np.ndarray
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
        self.batch = self.x.shape[0] if batch is None else batch
        self.weights = None
        self.losses = None

    def make_prediction(self, x):
        """
        makes prediction using a given features matrix and the weights determined from the Gradient descent
        algorithm.
        :param x: np.ndarray
            features matrix.
        :return: np.ndarray
            predicted values using the matrix and computed weights.
        """
        return x @ self.weights

    def grad_function(self, x, y, weights):
        """
        linear regression gradient of the loss function. This will be passed to the gradient descent
        algorithm to compute the gradient of the loss for each epoch(or iteration).
        :param x: np.ndarray
            features matrix for training
        :param y: np.ndarray
            target vector or matrix for training
        :param weights:
            computed weights to be evaluated
        :return: float
        """
        y = y.reshape((-1, 1))
        grad = 2 * x.T @ (x @ weights - y) / y.shape[0]
        return grad

    def loss_function(self, x, y, weights):
        """
        linear regression loss function. This will be passed to the gradient descent algorithm to compute
        the loss for each epoch.
        :param x: np.ndarray
            features matrix for training
        :param y: np.ndarray
            target vector or matrix for training
        :param weights:
            computed weights to be evaluated
        :return: float
        """
        y = y.reshape((-1, 1))
        return np.mean((y - (x @ weights)) ** 2)

    def fit(self):
        """
        Implements gradient descent algorithm to find the best weights for the model. This changes the weights
        and losses attributes.
        :return: None
        """
        grad_descent = GradientDescent(self.x, self.y, self.epochs, self.lr, self.batch, self.beta,
                                       self.loss_function, self.grad_function)
        grad_descent.fit()
        self.weights = grad_descent.weights
        self.losses = grad_descent.losses

    def r_square(self, x=None, y=None):
        """
        computes the r-square on the training set if no data is passed. When a dataset is passed,
        makes a prediction using x and the weights, then computes the r-square.
        :param x: np.ndarray | default = features matrix used for training
            features matrix for prediction.
        :param y: np.ndarray | default = target matrix used for training
            target matrix for evaluation.
        :return: float
        """
        x = self.x if x is not None else x
        y = self.y if y is not None else y
        y_pred = self.make_prediction(x)
        y_true = y
        y_mean = np.mean(y)
        ss_reg = np.sum((y_true - y_pred) ** 2)
        ss_total = np.sum((y_true - y_mean) ** 2)
        return 1 - ss_reg/ss_total

    def plot_function(self, x=None, y=None):
        """
        Plots the graphs of feature vector against target vector, and feature vector and predicted y.
        :param x: np.ndarray
            feature vector, should be a vector.
        :param y: np.ndarray
            target vector
        :return: graph
        """
        assert x.shape[1] == 1, "You can only plot one column feature matrix against a one column target matrix."
        x = self.x if x is not None else x
        y = self.y if y is not None else y
        y_pred = self.make_prediction(x)
        plt.figure(figsize=(12, 7))
        plt.plot(x, y, 'b*-', label='true_values')
        plt.plot(x, y_pred, 'r*-', label='Predicted values')
        plt.legend(loc='best')
        plt.show()
