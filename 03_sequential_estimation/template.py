from scipy._lib.doccer import extend_notes_in_docstring
from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(n: int, k: int, mean: np.ndarray, var: float) -> np.ndarray:
    """Generate n values samples from the k-variate
    normal distribution
    """
    arr = np.ndarray((n, k))
    cov = np.identity(k) * np.power(var, 2)
    for i in range(n):
        arr[i] = np.random.multivariate_normal(mean, cov)

    return arr


def update_sequence_mean(mu: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """Performs the mean sequence estimation update"""
    return mu + ((x - mu) / n)


def _plot_sequence_estimate():
    data = gen_data(100, 3, np.array([0, 0, 0]), 1)
    mean = np.mean(data, 0)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(mean, data[i], i + 1))
    print(estimates)
    plt.plot([e[0] for e in estimates], label="First dimension")
    plt.plot([e[1] for e in estimates], label="Second dimension")
    plt.plot([e[2] for e in estimates], label="Third dimension")

    plt.xlabel("Number of points")
    plt.ylabel("Mean")
    plt.title("Sequential estimation of mean vector")
    plt.legend(loc="upper center")
    plt.savefig("images/1_5_1.png")
    plt.show()


def _square_error(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))


def _plot_mean_square_error():
    data = gen_data(100, 3, np.array([0, 0, 0]), 1)
    mean = np.mean(data, 0)
    estimates = [np.array([0, 0, 0])]
    print(data.shape[0])
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(mean, data[i], data.shape[0]))
    square_errors = []
    print(len(estimates))
    for i in range(len(estimates)):
        square_errors.append(_square_error(mean, estimates[i]))
    plt.plot([e for e in square_errors])
    plt.xlabel("Number of points")
    plt.ylabel("Square error")
    plt.title("Square error of sequential mean estimates")
    plt.savefig("images/1_6_1.png")
    plt.show()


# Naive solution to the bonus question.


def gen_changing_data(
    n: int, k: int, start_mean: np.ndarray, end_mean: np.ndarray, var: float
) -> np.ndarray:
    # remove this if you don't go for the bonus
    mean_vector = np.linspace(start_mean, end_mean, 500)
    arr = np.ndarray((n, k))
    for i in range(n):
        arr[i, :] = gen_data(1, k, mean_vector[i], var)

    return arr


def running_mean(mu, n, old_x, new_x):
    return mu + ((new_x - old_x) / n)


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the bonus
    n = 500
    d = 3
    start_mean = [0, 1, -1]
    end_mean = [1, -1, 0]
    data = gen_changing_data(n, d, start_mean, end_mean, 1)
    mean_vec = np.linspace(start_mean, end_mean, n)
    window = 25
    estimates = np.zeros((n - window + 1, d))
    square_errors = np.zeros((n - window, d))
    estimates[0, :] = np.mean(data[0:window], 0)
    square_errors[0, :] = _square_error(estimates[0], mean_vec[0])
    for i in range(n - window):
        estimates[i + 1, :] = running_mean(
            estimates[i, :], window, data[i, :], data[i + window, :]
        )
        square_errors[i, :] = _square_error(estimates[i, :], mean_vec[i, :])

    plt.figure(1)
    plt.plot([e[0] for e in estimates], label="First dimension")
    plt.plot([e[1] for e in estimates], label="Second dimension")
    plt.plot([e[2] for e in estimates], label="Third dimension")
    plt.xlabel("Number of points")
    plt.ylabel("Mean")
    plt.title("Sequential estimation of changing mean vector")
    plt.legend(loc="upper center")
    plt.savefig("images/bonus_1.png")
    plt.show()

    plt.figure(2)
    plt.plot([s for s in square_errors])
    plt.xlabel("Number of points")
    plt.ylabel("Square error")
    plt.title("Square error of changing mean vector")
    plt.savefig("images/bonus_2.png")
    plt.show()


if __name__ == "__main__":
    # print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    # X = gen_data(300, 3, np.array([0, 1, -1]), np.sqrt(3))
    # scatter_3d_data(X)
    # bar_per_axis(X)
    # new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)

    # _plot_mean_square_error()
    X = gen_changing_data(100, 3, [0, 1, -1], [1, -1, 0], 1)
    # scatter_3d_data(X)
    # bar_per_axis(X)
    _plot_sequence_estimate()
    _plot_changing_sequence_estimate()
    _plot_mean_square_error()
