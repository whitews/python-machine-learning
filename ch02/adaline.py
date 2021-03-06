import numpy as np


class AdalineGD(object):
    """
    Adaptive linear neuron classifier

    Parameters
    ----------
    eta :
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
    cost_ : list
        Number of mis-classifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

        # these will be populated in fit
        self.w_ = None
        self.cost_ = None

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

        # reset cost
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(x)

            errors = (y - output)

            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = (errors**2).sum() / 2.0

            self.cost_.append(cost)

    def net_input(self, x):
        """Calculate net input"""
        net_input = np.dot(x, self.w_[1:]) + self.w_[0]
        return net_input

    def activation(self, x):
        """Compute linear activation"""
        return self.net_input(x)

    def predict(self, x):
        """Return class label after unit step"""
        prediction = np.where(self.activation(x) >= 0.0, 1, -1)
