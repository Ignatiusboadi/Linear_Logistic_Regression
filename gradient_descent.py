import numpy as np


class GradientDescent:
    """
    This class implements the gradient descent algorithm. Depending on the values passed, batch, mini-batch or
    stochastic gradient descent is implemented. If beta is 0, the algorithms are implemented without momentum, otherwise
    it is implemented with momentum.

    Attributes
    ----------
    lr: float
        outputs the learning rate used (to be used) in the algorithm
    beta: float
        outputs the value for beta (momentum) used.
    batch: int
        number of datasets to be used for each iteration in training the model. An output of None indicates batch
        gradient descent was used; 1 indicates stochastic gradient descent algorithm was implemented
    epochs: int
        outputs the number of epochs set to run in the implementation of the gradient descent algorithm.
    """
