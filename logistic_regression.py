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