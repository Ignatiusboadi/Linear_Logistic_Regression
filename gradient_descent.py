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
    n_features: int
        the number of features or columns of the input or features matrix.
    n_samples: int
        the number of rows or samples in the input or features matrix.
    weights: np.ndarray
        outputs the weights of the model after training.

    Methods
    -------
    initialize_weights:
        initilizes the weights before the training algorithm starts.

    """

    def __init__(self, x, y, epochs=1000, lr=0.1, batch=None, beta=0.9, loss_function=None, grad_function=None):
        """
        initializes an object of class GradientDescent. This will be used in implementing gradient descent algorithm.
        The following params can be specified:
        :param x: np.ndarray, pd.Series, pd.DataFrame.
                The features matrix to be used in the training of the weights of the model.
        :param y: np.ndarray, pd.Series, pd.DataFrame.
                The target matrix to be used in the training of the weights of the model.
        :param epochs: int | Optional | default = 1000
                The number of epochs.
        :param lr: 'float' | Optional | default = 0.1
                The learning rate.
        :param batch: int | Optional | default = None
                If batch is None, batch Gradient descent is implemented. If it is 1, stochastic gradient descent
                is implemented, otherwise minibatch is implemented with each batch containing the specified data
                points. Default is None, i.e, batch gradient descent.
        :param beta: float | Optional | default = 0.9
                specifies the momentum to be used in the training. Default is 0.9.
        :param loss_function: function
            The loss function to be used in the training. Must be specified.
        :param grad_function: function
            The gradient of the loss function to be used in training. Must be specified.
        """
        self.x = x
        self.y = y
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.beta = beta
        self.loss_function = loss_function
        self.grad_function = grad_function
        self.weights = None
        self.n_features = self.x.shape[1]
        self.n_samples = self.x.shape[0]

    def initialize_weights(self):
        """
        initializes the weights before the training algorithm starts. A 2-d array with one column and rows equal to
        the 'number of features + 1' is created. This sets the weight attribute to the weights created.
        :return: None.
        """
        self.weights = np.zeros((self.x.shape[1], 1))


