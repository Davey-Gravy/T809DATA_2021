from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.shape_base import split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    """
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    """
    result = np.ndarray(0)
    for i in range(len(classes)):
        count_train = 0
        for j in range(len(targets)):
            if targets[j] == classes[i]:
                count_train += 1
        result = np.append(result, count_train / len(targets))

    return result


def split_data(
    features: np.ndarray, targets: np.ndarray, split_feature_index: int, theta: float
) -> Union[tuple, tuple]:
    """
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    """
    features_1 = []
    features_2 = []
    targets_1 = []
    targets_2 = []
    for i in range(len(targets)):
        if features[i, split_feature_index] < theta:
            features_1.append(features[i])
            targets_1.append(targets[i])
        else:
            features_2.append(features[i])
            targets_2.append(targets[i])

    features_1 = np.array(features_1)
    features_2 = np.array(features_2)

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    """
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    """
    if len(targets) > 0:
        p_x = prior(targets, classes)
        sum = 0
        for i in range(len(classes)):
            sum += np.power(p_x[i], 2)
    else:
        return 1

    return 0.5 * (1 - sum)


def weighted_impurity(t1: np.ndarray, t2: np.ndarray, classes: list) -> float:
    """
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    """
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = len(t1) + len(t2)
    return (len(t1) * g1 / n) + (len(t2) * g2 / n)


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float,
) -> float:
    """
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    """
    (f1, t1), (f2, t2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t1, t2, classes)


def brute_best_split(
    features: np.ndarray, targets: np.ndarray, classes: list, num_tries: int
) -> Union[float, int, float]:
    """
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    """
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        thetas = np.linspace(features[:, i].min(), features[:, i].max(), num_tries + 2)[
            1:-1
        ]
        # iterate thresholds
        for theta in thetas:
            (f1, t1), (f2, t2) = split_data(features, targets, i, theta)
            impurity = weighted_impurity(t1, t2, classes)
            if impurity < best_gini:
                best_gini = impurity
                best_dim = i
                best_theta = theta

    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8,
    ):
        """
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        """
        (self.train_features, self.train_targets), (
            self.test_features,
            self.test_targets,
        ) = split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

    def plot(self):
        plt.figure(figsize=(10, 8))
        plot_tree(self.tree)
        plt.savefig("images/2_3_1.png")
        plt.show()

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        x = []
        y = []
        for i in range(self.train_features.shape[0]):
            self.tree.fit(self.train_features[0 : i + 1], self.train_targets[0 : i + 1])
            x.append(i)
            y.append(self.accuracy())
        plt.plot(x, y)
        plt.title("plot_progress()")
        plt.xlabel("Samples")
        plt.ylabel("Accuracy")
        plt.savefig("images/bonus_1.png")
        plt.show()

    def guess(self):
        return self.tree.predict(features)

    def confusion_matrix(self):
        n = len(self.classes)
        arr = np.zeros((n, n), dtype=int)
        pred = self.tree.predict(self.test_features)
        actual = self.test_targets

        np.add.at(arr, [actual, pred], 1)
        return arr.T


if __name__ == "__main__":

    # prior([0, 2, 3, 3], [0, 1, 2, 3])
    features, targets, classes = load_iris()
    # (f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
    # gini_impurity(t_1, classes)
    # gini_impurity(t_2, classes)
    # print(weighted_impurity(t_1, t_2, classes))
    # print(total_gini_impurity(features, targets, classes, 2, 4.65))
    # brute_best_split(features, targets, classes, 30)
    dt = IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
    dt.plot_progress()
    # dt.train()
    # dt.accuracy()
    # dt.guess()
    # dt.confusion_matrix()
    # dt.plot()
