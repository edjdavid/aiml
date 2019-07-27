import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

import sys
sys.path.append('../')

from aiml import KNNClassifier  # nopep8

cancer = load_breast_cancer()


def test_knn_classifier():
    rs = np.random.RandomState(1)
    train_acc = []
    test_acc = []
    for i in range(2):
        (X_train, X_test, y_train,
         y_test) = train_test_split(cancer.data, cancer.target, test_size=0.70,
                                    random_state=rs)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        train_acc.append(model.score(X_train, y_train))
        test_acc.append(model.score(X_test, y_test))

    knn_expected_train_mean = np.mean(train_acc)
    knn_expected_test_mean = np.mean(test_acc)
    knn_expected_train_std = np.std(train_acc)
    knn_expected_test_std = np.std(test_acc)

    knn = KNNClassifier([5])
    knn.n_trials = 2
    knn.random_state = 1
    knn.test_size = 0.70
    knn.train_test(cancer.data, cancer.target)
    assert knn_expected_train_mean == knn.training_accuracy
    assert knn_expected_test_mean == knn.test_accuracy
    assert knn_expected_train_std == knn.training_std
    assert knn_expected_test_std == knn.test_std