import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from aiml.models import KNNClassifier, LogisticRegressor, LinearSVM, MLModels

from numpy.testing import assert_equal

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
    assert_equal(knn_expected_train_mean, knn.training_accuracy)
    assert_equal(knn_expected_test_mean, knn.test_accuracy)
    assert_equal(knn_expected_train_std, knn.training_std)
    assert_equal(knn_expected_test_std, knn.test_std)


def test_logistic_classifier():
    lr_expected_train_mean = []
    lr_expected_test_mean = []
    lr_expected_train_std = []
    lr_expected_test_std = []
    rs = np.random.RandomState(1)
    C = [0.01]
    for c in C:
        train_acc = []
        test_acc = []
        for i in range(2):
            (X_train, X_test, y_train,
             y_test) = train_test_split(cancer.data, cancer.target,
                                        test_size=0.70, random_state=rs)
            model = LogisticRegression(C=c, solver='liblinear',
                                       penalty='l2', multi_class='auto')
            model.fit(X_train, y_train)

            train_acc.append(model.score(X_train, y_train))
            test_acc.append(model.score(X_test, y_test))

        lr_expected_train_mean.append(np.mean(train_acc))
        lr_expected_test_mean.append(np.mean(test_acc))
        lr_expected_train_std.append(np.std(train_acc))
        lr_expected_test_std.append(np.std(test_acc))

    lr = LogisticRegressor(C)
    lr.n_trials = 2
    lr.random_state = 1
    lr.test_size = 0.70
    lr.train_test(cancer.data, cancer.target)

    assert_equal(lr_expected_train_mean, lr.training_accuracy)
    assert_equal(lr_expected_test_mean, lr.test_accuracy)
    assert_equal(lr_expected_train_std, lr.training_std)
    assert_equal(lr_expected_test_std, lr.test_std)


def test_classifier_model_assignment():
    def noop(*args, **kwargs):
        pass

    MLModels.train_test = noop
    MLModels.plot_pcc = noop
    MLModels.summarize = noop

    models = MLModels.run_classifier(cancer.data, cancer.target,
                                     cancer.feature_names)
    assert isinstance(models['KNN'], KNNClassifier)
    assert isinstance(models['Logistic Regression (L1)'], LogisticRegressor)
    assert models['Logistic Regression (L1)'].model.penalty == 'l1'
    assert isinstance(models['Logistic Regression (L2)'], LogisticRegressor)
    assert models['Logistic Regression (L2)'].model.penalty == 'l2'
    assert isinstance(models['Linear SVM (L1)'], LinearSVM)
    assert models['Linear SVM (L1)'].model.penalty == 'l1'
    assert isinstance(models['Linear SVM (L2)'], LinearSVM)
    assert models['Linear SVM (L2)'].model.penalty == 'l2'

    models = MLModels.run_classifier(cancer.data, cancer.target,
                                     cancer.feature_names, methods='knn')
    assert set(models.keys()) == {'KNN'}

    models = MLModels.run_classifier(cancer.data, cancer.target,
                                     cancer.feature_names, methods='logistic')
    assert set(models.keys()) == {'Logistic Regression (L1)',
                                  'Logistic Regression (L2)'}

    models = MLModels.run_classifier(cancer.data, cancer.target,
                                     cancer.feature_names, methods='lr')
    assert set(models.keys()) == {'Logistic Regression (L1)',
                                  'Logistic Regression (L2)'}

    models = MLModels.run_classifier(cancer.data, cancer.target,
                                     cancer.feature_names, methods='svm')
    assert set(models.keys()) == {'Linear SVM (L1)',
                                  'Linear SVM (L2)'}

    models = MLModels.run_classifier(cancer.data, cancer.target,
                                     cancer.feature_names, methods='svc')
    assert set(models.keys()) == {'Linear SVM (L1)',
                                  'Linear SVM (L2)'}

    custom_methods = {
        'Nearest Neighbor': KNNClassifier([1]),
        'Linear SVM': LinearSVM([0.1], 'l2')
    }
    custom_models = MLModels.run_classifier(
        cancer.data, cancer.target, cancer.feature_names,
        methods=custom_methods)
    assert len(custom_models.keys()) == 2
    assert isinstance(custom_models['Nearest Neighbor'], KNNClassifier)
    assert isinstance(custom_models['Linear SVM'], LinearSVM)
    assert custom_models['Linear SVM'].model.penalty == 'l2'

    # not valid methods
    models = MLModels.run_classifier(cancer.data, cancer.target,
                                     cancer.feature_names, methods='svv')
    assert models is None

    models = MLModels.run_classifier(cancer.data, cancer.target,
                                     cancer.feature_names, methods={})
    assert models is None
