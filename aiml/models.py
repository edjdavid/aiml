import numpy as np
import pandas as pd
import warnings

from collections.abc import Sequence
from tqdm.autonotebook import tqdm

# plotting
import matplotlib.pyplot as plt

# ML algo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.exceptions import ConvergenceWarning

# this shouldn't be a hard dependency
from IPython.core.display import display


class MLModels:
    """
    Creates a class MLModels. 

    Parameters
    ----------
    
    n_trials : int
        default : 50
        
    test_size : float
        default : 0.25
        
    random_state: int
        default : None
        
    pred_var_setting: float
        default : 0.01
        
        
    Methods
    --------
    plot_accuracy : Plots and returns model train and test accuracies
    train_test : Calculates the training and testing accuracy of the model
    run_classifier : Runs the specified classifier algorithms
     on the data provided
    plot_pcc : Calculates the Proportion Chance Criteria,
     Plots a bar chart of all classes
    run_regression : Runs the specified regression algorithms
     on the data provided
    summarize : Displays in a dataframe the best performance
     (highest accuracy) of the methods
    
    """

    # safe to change
    n_trials = 50
    test_size = 0.25
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
        self.classes = None
        self._setting = None

    def plot_accuracy(self):
        """
        Plots the train and test accuracy +- 1 standard deviation of the model.
        """
        fig, ax = plt.subplots()
        ax.plot(self._setting, self.training_accuracy,
                label="training accuracy")
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

    def train_test(self, X, y, scaler=None):
        """
        Calculates the training and testing accuracy of the model.

        Calculate mean accuracy for `self.n_trials` using the given parameter
        `self._settings`.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data
        y : array-like, shape = [n_samples]
            Target vector relative to X    
        scaler : object
            Scaling method to be applied to X
            default : None
        """
        train_accuracies = []
        test_accuracies = []
        if self.pred_var_setting is not None:
            self._setting = sorted(list(set(self._setting)
                                        .union({self.pred_var_setting})))

        rs = (np.random.RandomState(seed=self.random_state) if
              self.random_state else None)
        with tqdm(total=self.n_trials*len(self._setting)) as pb:
            for i in range(self.n_trials):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=rs)
                if scaler is not None:
                    # scale using the training set
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    # apply the training set scale to the test set
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {i + 1}')
                training_accuracy = []
                test_accuracy = []
                feature_coef = []
                for s in self._setting:
                    # build the model
                    self.model.__setattr__(self._setting_name, s)
                    clf = self.model
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
            try:
                self.classes = clf.classes_
            except AttributeError:
                pass

    @staticmethod
    def run_classifier(X, labels, feature_names=None, C=None,
                       n_neighbors=None, scaler=None, methods='all'):
        """
        Run classifier algorithms on the data provided.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data
        labels : array-like, shape = [n_samples]
            Target vector relative to X
        feature_names : list
            List of column names to include in the training of the model
        C : list
            List of values of C for Logistic Regression and SVC
            default : 1e-8, 1e-4, 1e-3, 0.1, 0.2, 0.4, 0.75, 
                     1, 1.5, 3, 5, 10, 15, 20, 100, 300, 1000, 5000
        n_neighbors : list
            List of values of number of neighbors for KNN
            default : 1 to 50      
        scaler : object
            Scaling method to be applied to X
            default : None
        methods: Union[str, dict]
            Dictionary of label -> MLModel to execute or a str corresponding
            to a pre-defined list.
            default : 'all'
            str options : 'knn', 'logistic' or 'lr',
             'svc' or 'svm', 'nsvc' or 'nsvm'
            
        Returns
        -------
        Dictionary of MLModels
        """
        C = [1e-8, 1e-4, 1e-3, 0.1, 0.2, 0.4, 0.75, 1, 1.5, 3, 5, 10, 15, 20,
             100, 300, 1000, 5000] if C is None else C
        n_nb = list(range(1, 51)) if n_neighbors is None else n_neighbors
        
        if isinstance(methods, str):
            algo = {methods} if methods != 'all' else {'knn', 'lr', 'svc'}
            methods = {}

            if algo.intersection({'knn'}):
                methods['KNN'] = KNNClassifier(n_nb)

            if algo.intersection({'logistic', 'lr'}):
                methods['Logistic Regression (L1)'] =\
                    LogisticRegressor(C, 'l1')
                methods['Logistic Regression (L2)'] =\
                    LogisticRegressor(C, 'l2')

            if algo.intersection({'svc', 'svm'}):
                methods['Linear SVM (L1)'] = LinearSVM(C, 'l1')
                methods['Linear SVM (L2)'] = LinearSVM(C, 'l2')

            if algo.intersection({'svc', 'svm'}):
                methods['Polynomial SVM'] = PolynomialSVM({'C': C})
                methods['RBF SVM'] = RadialBasisSVM({'gamma': C})

        if not methods:
            warnings.warn('methods is not a valid value')
            return

        MLModels.plot_pcc(labels)
        plt.show()
        return MLModels.__run_models(methods, X, labels, feature_names,
                                     scaler=scaler)

    @staticmethod
    def plot_pcc(labels):
        """
        Calculates and prints the Proportion Chance Criterion.

        Plots the frequency of each class as a bar chart.
    
        Parameters
        ----------
        labels : array-like, shape = [n_samples]
            Target vector relative to X
        """
        label, counts = np.unique(labels, return_counts=True)
        N = np.sum(counts)
        pcc = np.sum([(n/N)**2 for n in counts])
        fig, ax = plt.subplots()
        ax.bar(range(len(counts)), counts, tick_label=label)
        ax.set_title('PCC = %.2f (%.2f)' % (pcc, pcc*1.25))
        ax.set_xlabel('labels')
        ax.set_ylabel('frequency')

        return ax

    @staticmethod
    def run_regression(X, labels, feature_names=None, alpha=None, 
                       n_neighbors=None, scaler=None, algorithm=['all']):
        """
        Runs the specified algorithms on the data provided.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}
            Training data
        labels : array-like, shape = [n_samples]
            Target vector relative to X
        feature_names : 
        alpha : list
            List of values of alpha for Linear Regression
            default : 1e-12, 1e-10, 1e-8, 1e-4, 1e-3,0.1, 0.2, 0.4, 0.75,
             1, 1.5, 3, 5, 10, 15,  20
        n_neighbors : list
            List of values of number of neighbors for KNN
            default : 1 to 50      
        scaler : object
            Scaling method to be applied to X
            default : None
        algorithm : list
            default : 'all'
            options : 'knn', 'linear' or 'linear regression'
            
        Returns
        -------
        Dictionary of model objects
        """
        alpha = [1e-12, 1e-10, 1e-8, 1e-4, 1e-3,0.1, 0.2, 0.4, 0.75,
                 1, 1.5, 3, 5, 10, 15,  20] if alpha is None else alpha
        n_nb = list(range(1, 51)) if n_neighbors is None else n_neighbors

        methods = {}
        
        if isinstance(algorithm, list):        
            for algo in algorithm:
                algo = algo.lower()
                if algo == 'knn' or algo == 'all':
                    methods['KNN'] = KNNRegressor(n_nb)
                if (algo == 'linear' or algo == 'linear regression' or algo
                        == 'all'):
                    methods['Linear Regression (L1)'] = LassoRegressor(
                        alpha=alpha)
                    methods['Linear Regression (L2)'] = RidgeRegressor(
                        alpha=alpha)
                if (algo == 'linear1' or algo == 'linear regression1' or algo
                        == 'all'):
                    methods['Linear Regression (L1)'] = LassoRegressor(
                        alpha=alpha)
                if (algo == 'linear2' or algo == 'linear regression2' or algo
                        == 'all'):
                    methods['Linear Regression (L2)'] = RidgeRegressor(
                        alpha=alpha)
                if algo not in ['all', 'knn', 'linear', 'linear regression']:
                    print(f'method {algo} not in options')

            return MLModels.__run_models(methods, X, labels, feature_names,
                                         scaler=scaler)
        else:
            print('Algorithms should be in a list')
            
    @staticmethod
    def __run_models(methods, X, labels, feature_names, scaler=None):
        """
        Perform the models on X and return a summary.

        Parameters
        ----------
        methods: dictionary
            Dictionary of objects (models)
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data
        labels : array-like, shape = [n_samples]
            Target vector relative to X
        feature_names : list
            List of column names to include in the training of the model     
        scaler : object
            Scaling method to be applied to X
            default : None
            
        Returns
        -------
        Dictionary of fitted classifiers       
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for k in methods:
                print(k)
                m = methods[k]
                m.train_test(X, labels, scaler=scaler)

        display(MLModels.summarize(methods, (
            feature_names if feature_names is not None else X.columns)))
        return methods

    @staticmethod
    def summarize(methods, feature_names):
        """
        Displays the best performance (highest accuracy) of the methods

        Parameters
        ----------
        methods: dictionary
            Dictionary of objects (models)
        feature_names : list
            List of column names to include in the training of the model     
            
        Returns
        -------
        Dataframe of the best performance (highest accuracy) of the methods
        specified along with the best parameter and top predictor
        """        
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
                # regressors doesn't have classes
                if m.classes is None:
                    features.append(
                        f'{feature_names[np.argmax(np.abs(m.coef))]}')
                    continue

                tp = np.unravel_index(np.argmax(np.abs(m.coef)), m.coef.shape)
                if m.coef.shape[0] == 1:
                    features.append(f'{feature_names[tp[1]]}')
                else:
                    features.append(
                        f'Class: {m.classes[tp[0]]}; {feature_names[tp[1]]}')
            else:
                features.append('N/A')

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
    model = KNeighborsClassifier()


class KNNRegressor(KNN):
    model = KNeighborsRegressor(algorithm='kd_tree')


class LinearRegressor(MLModels):
    model = None
    _setting_name = 'alpha'

    def __init__(self, alpha):
        super().__init__()
        self._setting = alpha


class LassoRegressor(LinearRegressor):
    model = Lasso(max_iter=10000)


class RidgeRegressor(LinearRegressor):
    model = Ridge()


class LinearClassifier(MLModels):
    model = None
    _setting_name = 'C'

    def __init__(self, C, reg='l2'):
        super().__init__()
        self._setting = C
        self._init_model(reg)

    def _init_model(self, reg):
        raise NotImplementedError()


class LogisticRegressor(LinearClassifier):
    def _init_model(self, reg):
        self.model = LogisticRegression(solver='liblinear', penalty=reg,
                                        multi_class='auto')


class LinearSVM(LinearClassifier):
    def _init_model(self, reg):
        self.model = LinearSVC(loss='squared_hinge',
                               dual=False, penalty=reg)


class NonLinearClassifier(MLModels):
    def __init__(self, hyperparameters):
        """

        :param hyperparameters: dict
        Dictionary of hyperparameter values. One of the values should be a
        list indicating the hyperparameter to scan.
        """
        super().__init__()
        hp_lst = [k for k, v in hyperparameters.items() if isinstance(v,
                                                                   Sequence)]
        assert (len(hp_lst) == 1), 'Hyperparameters should contain only 1 list'
        self._setting_name = hp_lst[0]
        self._setting = hyperparameters[self._setting_name]
        del hyperparameters[self._setting_name]
        self._init_model(hyperparameters)

    def _init_model(self, hyperparameters):
        raise NotImplementedError()


class NonLinearSVM(NonLinearClassifier):
    def _init_model(self, hyperparameters):
        def_hp = self._get_default_hyperparameters()
        if self._setting_name in def_hp:
            del def_hp[self._setting_name]
        def_hp.update(hyperparameters)
        self.model = SVC(**def_hp)

    def _get_default_hyperparameters(self):
        raise NotImplementedError()


class PolynomialSVM(NonLinearSVM):
    def _get_default_hyperparameters(self):
        return {
            'kernel': 'poly',
            'degree': 2,
            'coef0': 10,
            'C': 1,
            'gamma': 'scale'
        }


class RadialBasisSVM(NonLinearSVM):
    def _get_default_hyperparameters(self):
        return {
            'kernel': 'rbf',
            'gamma': 10,
            'C': 1
        }
