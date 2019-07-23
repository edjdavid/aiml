import numpy as np
import pandas as pd
import warnings

from tqdm.autonotebook import tqdm
from functools import partial

# plotting
import matplotlib.pyplot as plt

# ML algo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning


# Note: Someone please write the documentation ;)
class MLModels:
    # safe to change
    n_trials = 50
    random_state = None
    pred_var_setting = 0.01

    # not so safe to change
    model = None
    _setting_name = None

    def __init__(self):
        self.training_accuracy = None
        self.test_accuracy = None
        self.training_std = None
        self.test_std = None
        self.coef = None
        self._setting = None

    def plot_accuracy(self):
        fig, ax = plt.subplots()
        ax.plot(self._setting, self.training_accuracy, label="training accuracy")
        ax.plot(self._setting, self.test_accuracy, label="test accuracy")
        ax.fill_between(self._setting,
                        self.training_accuracy-self.training_std,
                        self.training_accuracy+self.training_std, alpha=0.2)
        ax.fill_between(self._setting, self.test_accuracy-self.test_std,
                        self.test_accuracy+self.test_std, alpha=0.2)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel(self._setting_name)
        ax.legend()
        return ax

    def train_test(self, X, y):
        train_accuracies = []
        test_accuracies = []
        if self.pred_var_setting is not None:
            self._setting = sorted(list(set(self._setting)
                                        .union({self.pred_var_setting})))
        with tqdm(total=self.n_trials*len(self._setting)) as pb:
            for i in range(self.n_trials):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=self.random_state)
                pb.set_description(f'Iter: {i + 1}')
                training_accuracy = []
                test_accuracy = []
                feature_coef = []
                for s in self._setting:
                    # build the model
                    clf = self.model(**{self._setting_name: s})
                    clf.fit(X_train, y_train)
                    # record training set accuracy
                    training_accuracy.append(clf.score(X_train, y_train))
                    # record generalization accuracy
                    test_accuracy.append(clf.score(X_test, y_test))
                    if s == self.pred_var_setting:
                        try:
                            feature_coef.append(clf.coef_)
                        except AttributeError:
                            pass
                    pb.update(1)

                train_accuracies.append(training_accuracy)
                test_accuracies.append(test_accuracy)

        self.training_accuracy = np.mean(train_accuracies, axis=0)
        self.test_accuracy = np.mean(test_accuracies, axis=0)
        self.training_std = np.std(train_accuracies, axis=0)
        self.test_std = np.std(test_accuracies, axis=0)
        if feature_coef:
            self.coef = np.mean(feature_coef, axis=0)

    def run_classifier(X, labels, feature_names=None, C=None,
                       n_neighbors=None):
        C = [1e-8, 1e-4, 1e-3, 0.1, 0.2, 0.4, 0.75, 1, 1.5, 3, 5, 10, 15, 20,
             100, 300, 1000, 5000] if C is None else C
        n_nb = list(range(1, 51)) if n_neighbors is None else n_neighbors
        methods = {
            'KNN': KNNClassifier(n_nb),
            'Logistic Regression (L1)': LogisticRegressor(C, 'l1'),
            'Logistic Regression (L2)': LogisticRegressor(C, 'l2'),
            'SVC (L1)': LinearSVM(C, 'l1'),
            'SVC (L2)': LinearSVM(C, 'l2')
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for k in methods:
                print(k)
                m = methods[k]
                m.train_test(X, labels)

        print(MLModels.summarize(methods, (
            feature_names if feature_names is not None else X.columns)))
        return methods

    def summarize(methods, feature_names=None):
        names = []
        accuracies = []
        parameters = []
        features = []
        for k in methods:
            m = methods[k]
            names.append(k)
            accuracies.append(np.max(m.test_accuracy))
            parameters.append('%s = %s' % (
                m._setting_name, m._setting[np.argmax(m.test_accuracy)]))
            if m.coef is not None:
                tp = np.unravel_index(np.argmax(m.coef), m.coef.shape)
                if tp[0] == 0:
                    features.append(f'{feature_names[tp[1]]}')
                else:
                    features.append(
                        f'Class {tp[0]}; {feature_names[tp[1]]}')
            else:
                features.append('None')

        return pd.DataFrame(zip(names, accuracies, parameters, features),
                            columns=['Model', 'Accuracy',
                                     'Best Parameter', 'Top Predictor'])


class KNN(MLModels):
    _setting_name = 'n_neighbors'
    pred_var_setting = None

    def __init__(self, neighbor_setting):
        super().__init__()
        self._setting = neighbor_setting


class KNNClassifier(KNN):
    model = KNeighborsClassifier


class KNNRegressor(KNN):
    model = partial(KNeighborsRegressor, algorithm='kd_tree')


class LinearRegressor(MLModels):
    model = None
    _setting_name = 'alpha'

    def __init__(self, alpha):
        super().__init__()
        self._setting = alpha


class LassoRegressor(LinearRegressor):
    model = partial(Lasso, max_iter=10000)


class RidgeRegressor(LinearRegressor):
    model = Ridge


class LinearClassifier(LinearRegressor):
    model = None
    _setting_name = 'C'

    def __init__(self, C, reg='l2'):
        self._setting = C
        self._init_model(reg)

    def _init_model(self, reg):
        raise NotImplementedError()


class LogisticRegressor(LinearClassifier):
    def _init_model(self, reg):
        self.model = partial(LogisticRegression,
                             solver='liblinear', penalty=reg,
                             multi_class='auto')


class LinearSVM(LinearClassifier):
    def _init_model(self, reg):
        self.model = partial(LinearSVC, loss='squared_hinge',
                             dual=False, penalty=reg)