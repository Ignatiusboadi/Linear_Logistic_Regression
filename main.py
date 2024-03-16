from logistic_regression import LogisticRegression
from linear_regression import LinearRegression

import numpy as np

np.random.seed(0)


def train_test_split(X, y):
    np.random.seed(0)
    train_size = 0.8
    n = int(len(X) * train_size)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_idx = indices[: n]
    test_idx = indices[n:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, y_train, X_test, y_test


# Generate data for linear regression implementation
def generate_data(n=1000):
    np.random.seed(0)
    x = np.linspace(-5.0, 5.0, n).reshape(-1, 1)
    y = (29 * x + 30 * np.random.rand(n, 1)).squeeze().reshape((-1, 1))
    x = np.hstack((np.ones_like(x), x))
    return x, y


x, y = generate_data()
lin_x_train, lin_y_train, lin_x_test, lin_y_test = train_test_split(x, y)
print(f"x_train:{lin_x_train.shape}, y_train: {lin_y_train.shape}, x_test: {lin_x_test.shape}, "
      f"y_test: {lin_y_test.shape}")

lin_reg = LinearRegression(lin_x_train, lin_y_train)
lin_reg.fit()
print('model weights', lin_reg.weights)
print('training r-square', lin_reg.r_square())
print('test r-square', lin_reg.r_square(lin_x_test, lin_y_test))
