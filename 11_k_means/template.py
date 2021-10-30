import numpy as np
import sklearn as sk
from statistics import mode
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(X: np.ndarray, Mu: np.ndarray) -> np.ndarray:
    """
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    """
    distances = np.zeros((X.shape[0], Mu.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Mu.shape[0]):
            distances[i, j] = np.sqrt(np.sum(np.power(X[i, :] - Mu[j, :], 2)))
    return distances


def determine_r(dist: np.ndarray) -> np.ndarray:
    """
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    """
    r = np.zeros(dist.shape)
    for i in range(dist.shape[0]):
        index = np.where(dist[i, :] == dist[i, :].min())[0]
        r[i, index[0]] = 1
    return r


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    """
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    """
    sum = 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            sum += R[i, j] * dist[i, j]
    return sum / R.shape[0]


def update_Mu(Mu: np.ndarray, X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    """
    out = np.zeros(Mu.shape)
    for i in range(X.shape[0]):
        for j in range(Mu.shape[0]):
            out[j, :] += R[i, j] * X[i, :] / np.sum(R[:, j])
    return out


def k_means(X: np.ndarray, k: int, num_its: int) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X - X_mean) / X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0:k], :]
    J = []
    for i in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        J.append(determine_j(R, dist))
        Mu = update_Mu(Mu, X_standard, R)

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, J


def _plot_j():
    plt.figure()
    plt.plot(range(len(J)), J)
    plt.title("Objective function value $\hat{J}$ vs. number of iterations")
    plt.xlabel("Number of iterations")
    plt.ylabel("$\hat{J}$")
    plt.savefig("images/1_6_1.png")
    plt.show()


def _plot_multi_j():
    plt.subplots(2, 2)
    plt.suptitle("Effect of $k$ on objective function $\hat{J}$")
    i = 0
    for k in [2, 3, 5, 10]:
        i += 1
        Mu, R, J = k_means(X, k, 10)
        plt.subplot(2, 2, i)
        plt.plot(range(10), J)
        plt.title(f"$k$ = {k}")
        plt.xlabel("Number of iterations")
        plt.ylabel("$\hat{J}$")
    plt.savefig("images/1_7_1.png")
    plt.show()


def k_means_predict(
    X: np.ndarray, t: np.ndarray, classes: list, num_its: int
) -> np.ndarray:
    """
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    """
    Mu, R, J = k_means(X, 3, num_its)
    clusters = [int(np.where(R[i, :] == 1)[0]) for i in range(len(t))]
    modes = []
    preds = []
    for i in classes:
        counts = []
        for j in range(len(t)):
            if i == t[j]:
                counts.append(clusters[j])
        modes.append(mode(counts))
    for j in range(len(t)):
        for i in classes:
            if clusters[j] == modes[i]:
                preds.append(i)
    return preds


def _iris_kmeans_accuracy():
    return confusion_matrix(y, preds), accuracy_score(y, preds)


def _my_kmeans_on_image():
    # image, (w, h) = image_to_numpy("images/buoys.png")
    # print(image)
    # Mu, R, J = k_means(image, 7, 5)
    ...


def plot_image_clusters():
    """
    Plot the clusters found using sklearn k-means.
    """
    image, (w, h) = image_to_numpy("images/buoys.png")
    count = 1
    for num_clusters in [2, 5, 10, 20]:
        km = KMeans(n_clusters=num_clusters)
        km.fit(image)
        plt.suptitle(f"Number of clusters = {num_clusters}")
        plt.subplot("121")
        plt.imshow(image.reshape(w, h, 3))
        plt.subplot("122")
        # uncomment the following line to run
        plt.imshow(km.labels_.reshape(w, h), cmap="plasma")
        plt.savefig(f"images/2_1_{count}")
        count += 1


def _gmm_info():
    gmm = GaussianMixture(n_components=3).fit(X)
    return gmm.means_,gmm.covariances_, gmm.weights_


def _plot_gmm():
    gmm = GaussianMixture(n_components=3).fit(X)
    pred = gmm.predict(X)
    plot_gmm_results(X,pred,gmm.means_,gmm.covariances_)


if __name__ == "__main__":
    # a = np.array([[1, 0, 0], [4, 4, 4], [2, 2, 2]])
    # b = np.array([[0, 0, 0], [4, 4, 4]])
    # # distances = distance_matrix(a, b)
    # dist = np.array([[1, 2, 3], [0.3, 0.1, 0.2], [7, 18, 2], [2, 0.5, 7]])
    # R = determine_r(dist)
    # print(determine_j(R, dist))
    # X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    # Mu = np.array([[0.0, 0.5, 0.1], [0.8, 0.2, 0.3]])
    # R = np.array([[1, 0], [0, 1], [1, 0]])
    # update_Mu(Mu, X, R)
    X, y, c = load_iris()
    # Mu, R, J = k_means(X, 4, 10)
    # _plot_j()
    # _plot_multi_j()
    # preds = k_means_predict(X, y, c, 5)
    # print(_iris_kmeans_accuracy())
    # _my_kmeans_on_image()
    # plot_image_clusters()
    # print(_gmm_info())
    _plot_gmm()