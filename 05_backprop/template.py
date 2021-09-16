from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid of x
    """
    if x < -100:
        return 0
    else:
        return np.divide(1, 1 + np.exp(-x))


def d_sigmoid(x: float) -> float:
    """
    Calculate the derivative of the sigmoid of x.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def perceptron(x: np.ndarray, w: np.ndarray) -> Union[float, float]:
    """
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    """
    sum = 0
    for i in range(x.shape[0]):
        sum += x[i] * w[i]

    return sum, sigmoid(sum)


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    """
    z0 = np.append(1.0, x)
    y, a1, a2, z1 = np.array([]), np.array([]), np.array([]), np.array([1.0])

    for i in range(M):
        a1 = np.append(a1, perceptron(z0, W1[:, i])[0])
        z1 = np.append(z1, perceptron(z0, W1[:, i])[1])

    for i in range(K):
        a2 = np.append(a2, perceptron(z1, W2[:, i])[0])
        y = np.append(y, perceptron(z1, W2[:, i])[1])
    return y, z0, z1, a1, a2


def backprop(
    x: np.ndarray, target_y: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    """
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    delta_k = y - target_y
    delta_j = []
    for i in range(len(a1)):
        delta_j.append(d_sigmoid(a1[i]) * (np.sum(W2[i + 1] * delta_k)))

    dE1 = np.zeros(W1.shape)
    dE2 = np.zeros(W2.shape)

    for i in range(len(delta_j)):
        for j in range(len(z0)):
            dE1[j, i] = delta_j[i] * z0[j]

    for i in range(len(delta_k)):
        for j in range(len(z1)):
            dE2[j, i] = delta_k[i] * z1[j]

    return y, dE1, dE2


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    """
    ...


def test_nn(
    X: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray
) -> np.ndarray:
    """
    Return the predictions made by a network for all features
    in the test set X.
    """
    ...


if __name__ == "__main__":
    # print(sigmoid(0.5))
    # print(d_sigmoid(0.2))
    print(perceptron(np.array([1.0, 2.3, 1.9]), np.array([0.2, 0.3, 0.1]))[0])
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(
        features, targets
    )
    D = train_features.shape[0]
    x = train_features[0, :]
    K = 3
    M = 10
    # initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    print(ffnn(x, M, K, W1, W2))
