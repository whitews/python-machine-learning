import numpy as np


class Perceptron(object):
    """
    Perceptron classifier

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Number of passes over the training data set

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        Number of mis-classifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

        # these will be populated in fit
        self.w_ = None
        self.errors_ = None

    def fit(self, x, y):
        """
        Fit training data

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features
        y : array-like, shape = [n_samples]
            Target values

        Returns
        -------
        None
        """

        # set weights to zero-vector of length n_features + 1,
        # the extra one is for the zero-weight (i.e. the threshold)
        self.w_ = np.zeros(1 + x.shape[1])

        # reset errors
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))

                self.w_[1:] += update * xi
                self.w_[0] += update

                errors += int(update != 0.0)

            self.errors_.append(errors)

    def net_input(self, x):
        """Calculate net input"""
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        """Return class label after unit step"""
        return np.where(self.net_input(x) >= 0.0, 1, -1)
