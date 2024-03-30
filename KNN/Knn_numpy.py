# coding: utf-8
"""
    Задание: ближайшие соседи, синтетический датасет, Евклидово расстояние.

    В задании требуется реализовать метод ближайших соседей для двух классов
    с параметризуемым числом соседей.
"""

import numpy as np
from typing import SupportsIndex
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score


class KNN(object):
    """
        Класс с реализацией метода ближайших соседей.
    """

    def __init__(self, n_neighbours: int = 4):
        # обучающая выборка: признаки
        self.X_train = None

        # обучающая выборка: метки классов
        self.y_train = None

        # число ближайших соседей
        self.n_neighbours = n_neighbours

    def fit(self, X: np.ndarray, y: SupportsIndex):
        """
            В методе fit (по аналогии с API sklearn) происходит обучение модели.
            Здесь как такового обучения у нас нет, надо просто запомнить датасет
            как "состояние объекта" KNN.
        """
        self.X_train = X
        self.y_train = y

        pass

    def predict(self, X: np.ndarray) -> np.ndarray:

        axes_dist = (X.reshape(X.shape[0], 1, 2) -
                     self.X_train.reshape(1, self.X_train.shape[0], 2))
        # Расстояния между всеми парами точек:
        dist = np.sqrt(np.sum(np.square(axes_dist), axis=2))

        sorted_dist = np.argsort(dist, axis=1, kind="quicksort")

        neighbours_in_1 = np.sum(self.y_train[sorted_dist[:, :self.n_neighbours]], axis=1)

        prediction = (neighbours_in_1 * 2 >= self.n_neighbours)
        return prediction


def accuracy(labels_true: np.ndarray, labels_predicted: np.ndarray):
    """
    :param labels_true: одномерный массив int-ов, истинные метки
    :param labels_predicted: одномерный массив int-ов, предсказанные метки
    :return: число совпавших меток делим на общее число меток
    """

    total_matches = np.sum(labels_true == labels_predicted)
    size = labels_true.size
    return total_matches / size


if __name__ == "__main__":

    np.random.seed(104)

    means0 = [-30, -1]
    covs0 = [[1710, 100.8],
             [100.8, 30.1]]
    x0, y0 = np.random.multivariate_normal(means0, covs0, 190).T

    means1 = [0, -1]
    covs1 = [[1510, 0.0],
             [0.0, 1200]]
    x1, y1 = np.random.multivariate_normal(means1, covs1, 100).T

    data0 = np.vstack([x0, y0]).T
    labels0 = np.zeros(data0.shape[0])

    data1 = np.vstack([x1, y1]).T
    labels1 = np.ones(data1.shape[0])

    data = np.vstack([data0, data1])
    labels = np.hstack([labels0, labels1])
    total_size = data.shape[0]
    print("Original dataset shapes:", data.shape, labels.shape)

    train_size = int(total_size * 0.7)
    indices = np.random.permutation(total_size)

    X_train, y_train = data[indices][:train_size], labels[indices][:train_size]
    X_test, y_test = data[indices][train_size:], labels[indices][train_size:]
    print("Train/test sets shapes:", X_train.shape, X_test.shape)


    for n_neighbours in range(1,6):
        predictor = KNN(n_neighbours)

        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)

        if n_neighbours == 1:
            y_pred1 = y_pred
        if n_neighbours == 3:
            y_pred3 = y_pred
        if n_neighbours == 5:
            y_pred5 = y_pred

        print("n_neighbours = %.1u:" % n_neighbours)

        print("Accuracy: %.4f [ours]" % accuracy(y_test, y_pred))

        print("Accuracy: %.4f [sklearn]" % accuracy_score(y_test, y_pred))

    plt.title('y_test: исходная выборка')
    plt.plot(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1],
             color='b', marker='o', markersize=5, ls="")
    plt.plot(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
             color='r', marker='o', markersize=5, ls="")
    plt.plot(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
             color='b', marker='x', markersize=5, ls="", alpha=0.4)
    plt.plot(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
             color='r', marker='x', markersize=5, ls="", alpha=0.4)
    # plt.show()
    plt.savefig('Plot_y_test.png')
    plt.clf()

    plt.title('n_neighbours = 1')
    plt.plot(X_test[y_pred1 == 0][:, 0], X_test[y_pred1 == 0][:, 1],
             color='b', marker='o', markersize=5, ls="")
    plt.plot(X_test[y_pred1 == 1][:, 0], X_test[y_pred1 == 1][:, 1],
             color='r', marker='o', markersize=5, ls="")
    plt.plot(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
             color='b', marker='x', markersize=5, ls="", alpha=0.4)
    plt.plot(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
             color='r', marker='x', markersize=5, ls="", alpha=0.4)
    # plt.show()
    plt.savefig('Plot_n_neighbours_1.png')
    plt.clf()

    plt.title('n_neighbours = 3')
    plt.plot(X_test[y_pred3 == 0][:, 0], X_test[y_pred3 == 0][:, 1],
             color='b', marker='o', markersize=5, ls="")
    plt.plot(X_test[y_pred3 == 1][:, 0], X_test[y_pred3 == 1][:, 1],
             color='r', marker='o', markersize=5, ls="")
    plt.plot(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
             color='b', marker='x', markersize=5, ls="", alpha=0.4)
    plt.plot(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
             color='r', marker='x', markersize=5, ls="", alpha=0.4)
    # plt.show()
    plt.savefig('Plot_n_neighbours_3.png')
    plt.clf()

    plt.title('n_neighbours = 5')
    plt.plot(X_test[y_pred5 == 0][:, 0], X_test[y_pred5 == 0][:, 1],
             color='b', marker='o', markersize=5, ls="")
    plt.plot(X_test[y_pred5 == 1][:, 0], X_test[y_pred5 == 1][:, 1],
             color='r', marker='o', markersize=5, ls="")
    plt.plot(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
             color='b', marker='x', markersize=5, ls="", alpha=0.4)
    plt.plot(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
             color='r', marker='x', markersize=5, ls="", alpha=0.4)
    # plt.show()
    plt.savefig('Plot_n_neighbours_5.png')
    plt.clf()
