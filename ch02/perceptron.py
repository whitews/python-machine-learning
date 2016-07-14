import numpy as np
import matplotlib.pyplot as plt


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
                prediction = self.predict(xi)
                update = self.eta * (target - prediction)

                self.w_[1:] += update * xi
                self.w_[0] += update

                errors += int(update != 0.0)

                if update != 0.0:
                    # plot updated line
                    # Note: only works for data sets with 2 features!
                    line_x = np.linspace(x[:, 0].min(), x[:, 0].max(), 2)
                    line_y = (self.w_[1] / (-1 * self.w_[2]) * line_x) + self.w_[0] / (-1 * self.w_[2])

                    plt.plot(line_x, line_y)

                    plt.show()

            self.errors_.append(errors)

            if errors == 0:
                break

    def net_input(self, xi):
        """Calculate net input"""
        net_input = np.dot(xi, self.w_[1:]) + self.w_[0]
        return net_input

    def predict(self, xi):
        """Return class label after unit step"""
        prediction = np.where(self.net_input(xi) >= 0.0, 1, -1)
        return prediction
