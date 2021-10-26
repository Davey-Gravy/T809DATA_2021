import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import _m_broadcast_to

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(features: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    """
    arr = np.zeros((features.shape[0], mu.shape[0]))
    for i in range(features.shape[0]):
        for j in range(mu.shape[0]):
            arr[i, j] = multivariate_normal.pdf(
                features[i, :], mean=mu[j, :], cov=sigma
            )
    return arr


def _plot_mvn(fi):
    print(fi.shape)
    plt.figure()
    for i in range(fi.shape[1]):
        # print(fi[:, i])
        plt.plot(range(fi.shape[0]), fi[:, i], label=i)
    plt.title("Output of basis functions as a function of features")
    plt.xlabel("x")
    plt.ylabel("$\phi(x)$")
    plt.savefig("images/1_2_1.png")
    plt.show()


def max_likelihood_linreg(
    fi: np.ndarray, targets: np.ndarray, lamda: float
) -> np.ndarray:
    """
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    """
    K = fi @ fi.T
    a = np.linalg.inv(K + lamda * np.identity(fi.shape[0])) @ t
    return fi.T @ a


def linear_model(
    features: np.ndarray, mu: np.ndarray, sigma: float, w: np.ndarray
) -> np.ndarray:
    """
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    """
    fi = mvn_basis(features, mu, sigma)
    print(fi.shape)
    print(w.shape)
    return w @ mvn_basis(features, mu, sigma).T


if __name__ == "__main__":
    X, t = load_regression_iris()
    N, D = X.shape
    M, sigma, i = 10, 10, 0
    mu = np.zeros((M, D))
    while i < D:
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
        i += 1
    fi = mvn_basis(X, mu, sigma)  # same as before
    _plot_mvn(fi)
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)
    print(linear_model(X, mu, sigma, wml))
