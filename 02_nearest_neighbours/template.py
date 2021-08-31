import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points
import help


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the euclidian distance between points x and y
    """
    sum = 0
    for i in range(len(x)):
        sum += np.power(x[i] - y[i], 2)

    return np.sqrt(sum)


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Calculate the euclidian distance between x and and many
    points
    """
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])

    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    """
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    """
    # find distances between x's and points, return the indices of first k points
    return np.argsort(euclidian_distances(x, points))[0:k]


def vote(targets, classes):
    """
    Given a list of nearest targets, vote for the most
    popular
    """
    # print(Counter(targets).most_common(1)[0][0])
    # return Counter(targets).most_common(1)[0][0]
    counts = np.bincount(targets)
    return np.argmax(counts)


def knn(
    x: np.ndarray, points: np.ndarray, point_targets: np.ndarray, classes: list, k: int
) -> np.ndarray:
    """
    Combine k_nearest and vote
    """
    # print(vote(k_nearest(x, points, k), classes))
    # print(point_targets)
    return vote(point_targets[k_nearest(x, points, k)], classes)


def knn_predict(
    points: np.ndarray, point_targets: np.ndarray, classes: list, k: int
) -> np.ndarray:
    results = np.ndarray(0, dtype=int)
    for i in range(points.shape[0]):
        x = points[i, :]
        points_i = help.remove_one(points, i)
        point_targets_i = help.remove_one(point_targets, i)
        results = np.append(results, knn(x, points_i, point_targets_i, classes, k))
    return results


def knn_accuracy(
    points: np.ndarray, point_targets: np.ndarray, classes: list, k: int
) -> float:
    count = 0
    prediction = knn_predict(points, point_targets, classes, k)
    for i in range(len(prediction)):
        if prediction[i] != point_targets[i]:
            count += 1
    return 1 - (count / len(prediction))


def knn_confusion_matrix(
    points: np.ndarray, point_targets: np.ndarray, classes: list, k: int
) -> np.ndarray:
    n = len(classes)
    arr = np.zeros((n, n), dtype=int)
    prediction = knn_predict(points, point_targets, classes, k)
    actual = point_targets

    np.add.at(arr, (actual, prediction), 1)
    return arr.T


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    best = 0
    accuracy = 0
    for k in range(1, len(points) - 1):
        if knn_accuracy(points, point_targets, classes, k) > accuracy:
            accuracy = knn_accuracy(points, point_targets, classes, k)
            best = k

    return best


def knn_plot_points(
    points: np.ndarray, point_targets: np.ndarray, classes: list, k: int
):
    edgecolor = []
    prediction = knn_predict(points, point_targets, classes, k)
    for i in range(len(prediction)):
        if prediction[i] != point_targets[i]:
            edgecolor.append("red")
        else:
            edgecolor.append("green")
    colors = ["yellow", "purple", "blue"]
    for i in range(points.shape[0]):
        [x, y] = points[i, :2]
        plt.scatter(
            x, y, c=colors[point_targets[i]], edgecolors=edgecolor[i], linewidths=2
        )
    plt.title("Yellow=0, Purple=1, Blue=2")
    plt.savefig("images/2_5_1.png")
    plt.show()


def weighted_vote(targets: np.ndarray, distances: np.ndarray, classes: list) -> int:
    """
    Given a list of nearest targets, vote for the most
    popular
    """
    # Remove if you don't go for independent section
    weights = np.divide(targets, distances)
    return np.argmax(weights)


def wknn(
    x: np.ndarray, points: np.ndarray, point_targets: np.ndarray, classes: list, k: int
) -> np.ndarray:
    """
    Combine k_nearest and vote
    """
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray, point_targets: np.ndarray, classes: list, k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(points: np.ndarray, targets: np.ndarray, classes: list):
    # Remove if you don't go for independent section
    ...


if __name__ == "__main__":
    d, t, classes = load_iris()
    x, points = d[0, :], d[1:, :]
    x_target, point_targets = t[0], t[1:]
    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
    # euclidian_distance(x, points[50])
    # print(euclidian_distances(x,points))
    # print(k_nearest(x, points, 3))
    # print(vote(np.array([0, 1, 1, 2]), np.array([0, 1, 2])))
    # print(point_targets)
    # print(knn(x, points, point_targets, classes, 5))
    # print(point_targets[10])
    # print(knn_predict(d_test, t_test, classes, 10))
    # print(knn_accuracy(d_test, t_test, classes, 10))
    # print(knn_confusion_matrix(d_test, t_test, classes, 10))
    # print(knn_accuracy(d_test, t_test, classes, 5))
    # print(knn_accuracy(d_test, t_test, classes, 118))
    # print(best_k(d_train, t_train, classes))
    # knn_plot_points(d, t, classes, 3)
    weighted_vote()
