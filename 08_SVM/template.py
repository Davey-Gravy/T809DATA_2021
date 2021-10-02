from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt


def _plot_linear_kernel():
    X, t = make_blobs(40, 2, 2)
    clf = svm.SVC(C=1000, kernel="linear")
    clf.fit(X, t)
    plot_svm_margin(clf, X, t)


def _subplot_svm_margin(svc, X: np.ndarray, t: np.ndarray, num_plots: int, index: int):
    """
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.

    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    """
    # similar to tools.plot_svm_margin but added num_plots and
    # index where num_plots should be the total number of plots
    # and index is the index of the current plot being generated
    if index == 1:
        plt.figure()
    plt.subplot(1, num_plots, index)
    plt.scatter(X[:, 0], X[:, 1], c=t, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svc.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )

    # plot support vectors
    ax.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )


def _compare_gamma():
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(C=1000, kernel="rbf")
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 3, 1)
    plt.gca().set_title("Default gamma")

    clf = svm.SVC(C=1000, kernel="rbf", gamma=0.2)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 3, 2)
    plt.gca().set_title("Gamma = 0.2")

    clf = svm.SVC(C=1000, kernel="rbf", gamma=2)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 3, 3)
    plt.gca().set_title("Gamma = 2")

    plt.suptitle("SVM with radial basis function")
    plt.savefig("images/1_3_1.png")
    plt.show()


def _compare_C():
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(C=1000, kernel="linear")
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 1)
    plt.gca().set_title("$C = 1000$")

    clf = svm.SVC(C=0.5, kernel="linear")
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 2)
    plt.gca().set_title("$C = 0.5$")

    clf = svm.SVC(C=0.3, kernel="linear")
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 3)
    plt.gca().set_title("$C = 0.3$")

    clf = svm.SVC(C=0.05, kernel="linear")
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 4)
    plt.gca().set_title("$C = 0.05$")

    clf = svm.SVC(C=0.001, kernel="linear")
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 5)
    plt.gca().set_title("$C = 0.001$")

    plt.suptitle("SVM with linear basis function")
    plt.savefig("images/1_5_1.png")
    plt.show()


def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
    """
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    """
    svc.fit(X_train, t_train)
    y = svc.predict(X_test)
    return (
        accuracy_score(t_test, y),
        precision_score(t_test, y),
        recall_score(t_test, y),
    )


if __name__ == "__main__":
    # _plot_linear_kernel()
    # _compare_gamma()
    # _compare_C()'
    (X_train, t_train), (X_test, t_test) = load_cancer()
    svmLinear = svm.SVC(C=1000, kernel="linear")
    svmSigmoid = svm.SVC(C=1000, kernel="sigmoid")
    svmRBF = svm.SVC(C=1000, kernel="rbf")
    print(train_test_SVM(svmLinear, X_train, t_train, X_test, t_test))
    print(train_test_SVM(svmSigmoid, X_train, t_train, X_test, t_test))
    print(train_test_SVM(svmRBF, X_train, t_train, X_test, t_test))
