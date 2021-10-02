from typing import Union
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.numeric import tensordot
from sklearn.metrics import confusion_matrix, accuracy_score

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
    guesses = [0] * len(X_train)
    N = len(X_train)
    misclassification_rate, Etotal = [], []

    for i in range(iterations):
        dE1_total, dE2_total = np.zeros(W1.shape), np.zeros(W2.shape)

        err, misclass = 0, 0

        for j in range(N):
            target_y = np.zeros(K)
            target_y[t_train[j]] = 1.0

            y, dE1, dE2 = backprop(X_train[j], target_y, M, K, W1, W2)

            dE1_total += dE1
            dE2_total += dE2

            guesses[j] = np.argmax(y)

            err += (target_y * np.log(np.array(y))) + (
                (1 - target_y) * np.log(1 - np.array(y))
            )

            if np.argmax(target_y) != guesses[j]:
                misclass += 1

        W1 -= eta * dE1_total / N
        W2 -= eta * dE2_total / N

        Etotal.append(np.sum(-err) / N)
        misclassification_rate.append(misclass / N)

    return W1, W2, Etotal, misclassification_rate, guesses


def test_nn(
    X: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray
) -> np.ndarray:
    """
    Return the predictions made by a network for all features
    in the test set X.
    """
    guesses = []
    for x in X:
        y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
        guesses.append(np.argmax(y))
    return guesses


if __name__ == "__main__":
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(
        features, targets
    )
    D = train_features.shape[0]
    x = train_features[0, :]
    K = 3
    layer_size = []
    score = []
    for M in range(10, 100, 10):
        # initialize two random weight matrices
        W1 = 2 * np.random.rand(D + 1, M) - 1
        W2 = 2 * np.random.rand(M + 1, K) - 1
        iterations = 500
        W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
            train_features[:20, :], train_targets[:20], M, K, W1, W2, iterations, 0.3
        )
        guesses = test_nn(test_features, M, K, W1tr, W2tr)
        # conf_matrix = confusion_matrix(test_targets, guesses)
        layer_size.append(M)
        score.append(accuracy_score(test_targets, guesses))

    plt.figure()
    plt.plot(layer_size, score)

    # plt.figure(1)
    # plt.plot(range(iterations), Etotal)
    # plt.title("$E_{total}$ as a function of iterations")
    # plt.xlabel("Iterations")
    # plt.ylabel("Total error $E_{total}$")
    # plt.savefig("images/error.eps", bbox_inches="tight")

    # plt.figure(2)
    # plt.plot(range(iterations), misclassification_rate)
    # plt.title("Misclassification rate as a function of iterations")
    # plt.xlabel("Iterations")
    # plt.ylabel("Misclassification rate")
    # plt.savefig("images/misclassification.eps", bbox_inches="tight")

    # fig = plt.figure(3)
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(conf_matrix)
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         plt.text(
    #             x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="large"
    #         )
    # cbar = plt.colorbar(cax)
    # cbar.set_label("Number of guesses", rotation=270, labelpad=15)
    # plt.title("Confusion matrix for iris classification")
    # plt.xlabel("Prediction")
    # plt.ylabel("Actual")
    # plt.savefig("images/confusiocdn_matrix.eps", bbox_inches="tight")
