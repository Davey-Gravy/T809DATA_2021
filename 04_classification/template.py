from tools import load_iris, split_train_test
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray, targets: np.ndarray, selected_class: int
) -> np.ndarray:
    """
    Estimate the mean of a selected class given all features
    and targets in a dataset
    """
    arr = [features[i] for i in range(len(targets)) if selected_class == targets[i]]
    return np.mean(arr, 0)
    # return [np.mean(i) for i in zip(*arr)]


def covar_of_class(
    features: np.ndarray, targets: np.ndarray, selected_class: int
) -> np.ndarray:
    """
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    """
    arr = [features[i] for i in range(len(targets)) if selected_class == targets[i]]
    return np.cov(arr, rowvar=False)


def likelihood_of_class(
    feature: np.ndarray, class_mean: np.ndarray, class_covar: np.ndarray
) -> float:
    """
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    """
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list,
) -> np.ndarray:
    """
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    """
    means, covs = [], []
    # loop through classes, append means, covariances for each point for each class
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    # print(means)
    # print(covs)
    likelihoods = []
    # print(likelihoods)
    # loop through all test features
    for i in range(test_features.shape[0]):
        f = []
        # loop through each class, append likelihood of each class for each feature
        for c in classes:
            f.append(likelihood_of_class(test_features[i, :], means[c], covs[c]))
        likelihoods.append(f)

    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    """
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    """
    arr = []
    # loop through likelihoods of all features
    for i in range(len(likelihoods)):
        max = -np.inf
        best = -1
        likelihood = likelihoods[i]
        # loop through likelihood of each class for each feature
        # if likelihood of class is greater than max, append its index
        for j in range(len(likelihood)):
            if likelihood[j] > max:
                max = likelihood[j]
                best = j
        arr.append(best)
    return np.array(arr)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list,
) -> np.ndarray:
    """
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    """
    means, covs = [], []
    sum = np.zeros(len(classes))

    # loop through classes, append means, covariances for each point for each class
    for c in classes:
        means.append(mean_of_class(train_features, train_targets, c))
        covs.append(covar_of_class(train_features, train_targets, c))
        for j in range(train_targets.shape[0]):
            if train_targets[j] == c:
                sum[c] += 1
    probs = np.divide(sum, train_targets.shape[0])

    likelihoods = []
    # loop through all test features
    for i in range(test_features.shape[0]):
        f = []
        # loop through each class, append likelihood of each class for each feature
        for c in classes:
            f.append(likelihood_of_class(test_features[i, :], means[c], covs[c]))
        likelihoods.append(f)
    print(likelihoods)
    print(probs)

    return np.array(likelihoods * probs)


def confusion_matrix(
    train_features, train_targets, test_features, test_targets, classes
):
    n = len(classes)
    arr_l = np.zeros((n, n), dtype=int)
    arr_a = np.zeros((n, n), dtype=int)
    pred_max_l = predict(
        maximum_likelihood(train_features, train_targets, test_features, classes)
    )
    pred_max_a = predict(
        maximum_aposteriori(train_features, train_targets, test_features, classes)
    )
    actual = test_targets
    # print(pred_max_l)
    # print(pred_max_a)
    # print(actual)
    np.add.at(arr_l, (actual, pred_max_l), 1)
    np.add.at(arr_a, (actual, pred_max_a), 1)

    print(arr_l.T)
    print(arr_a.T)


if __name__ == "__main__":
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(
        features, targets, train_ratio=1
    )
    class_mean = mean_of_class(train_features, train_targets, 0)
    # print(class_mean)
    # print(covar_of_class(train_features, train_targets, 0))
    # print(likelihood_of_class(test_features[0, :], class_mean, class_covar))
    # print(maximum_likelihood(train_features, train_targets, test_features, classes))
    # likelihoods = maximum_likelihood(
    # train_features, train_targets, test_features, classes
    # )
    # print(predict(likelihoods))
    # maximum_aposteriori(train_features, train_targets, test_features, classes)
    confusion_matrix(
        train_features, train_targets, test_features, test_targets, classes
    )
