import numpy as np

class MinMaxScaler:

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        
        return self

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        """
        Fit using X and then transform it. Useful when we need to scale just once.
        """
        self.fit(X)
        return self.transform(X)