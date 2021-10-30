import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    """
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    """

    out = np.ndarray(X.shape)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(X.shape[0]):
        out[i] = np.divide((X[i] - mu), sigma)
    return out


def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    """
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    """
    for n in range(X.shape[0]):
        plt.scatter(X[n, i], X[n, j], c="C0")


def _scatter_cancer():
    X, y, features = load_cancer()
    X_hat = standardize(X)
    plt.figure(figsize=[16, 8])
    for i in range(X.shape[1]):
        plt.subplot(5, 6, i + 1)
        plt.title(f"{features[i]}")
        scatter_standardized_dims(X_hat, 0, i)
    plt.savefig("images/1_3_1.png")


def _plot_pca_components():
    X, y, features = load_cancer()
    X = standardize(X)
    pca = PCA(n_components=30)
    pca.fit_transform(X)
    plt.figure(figsize=[16, 8])
    plt.suptitle("PCA on breast cancer data set")
    for i in range(X.shape[1]):
        plt.subplot(5, 6, i + 1)
        plt.title(f"Component {i+1}")
        plt.plot(pca.components_[i, :])
    plt.savefig("images/2_1_1.png")
    plt.show()


def _plot_eigen_values():
    X, y, features = load_cancer()
    X = standardize(X)
    pca = PCA(n_components=30)
    pca.fit_transform(X)
    plt.plot(pca.explained_variance_)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.savefig("images/3_1_1.png")
    plt.show()


def _plot_log_eigen_values():
    X, y, features = load_cancer()
    X = standardize(X)
    pca = PCA(n_components=30)
    pca.fit_transform(X)
    plt.plot(np.log10(pca.explained_variance_))
    plt.xlabel("Eigenvalue index")
    plt.ylabel("$\log_{10}$ Eigenvalue")
    plt.grid()
    plt.savefig("images/3_2_1.png")
    plt.show()


def _plot_cum_variance():
    X, y, features = load_cancer()
    X = standardize(X)
    pca = PCA(n_components=30)
    pca.fit_transform(X)
    plt.plot(np.cumsum(pca.explained_variance_) / np.sum(pca.explained_variance_))
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Percentage variance")
    plt.grid()
    plt.savefig("images/3_3_1.png")
    plt.show()


if __name__ == "__main__":
    # X = np.array([[1, 2, 3, 4], [0, 0, 0, 0], [4, 5, 5, 4], [2, 2, 2, 2], [8, 6, 4, 2]])
    # scatter_standardized_dims(X, 0, 2)
    # plt.show()
    # _scatter_cancer()
    # _plot_pca_components()
    # _plot_eigen_values()
    # _plot_log_eigen_values()
    _plot_cum_variance()
