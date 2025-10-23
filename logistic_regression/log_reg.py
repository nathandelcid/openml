import numpy as np

from sklearn.metrics import accuracy_score

class LogisticRegression:

    def __init__(self, eta=0.1, alpha=0):
        """
        Create a logistic regression classifier
        :param eta: Learning rate
        :param alpha: We will use this parameter later (IN BONUS)
        """

        self.w = None  # uninitialized w
        self.eta = eta  # learning rate
        self.initialized = False  # flag used to initialize w only once, it allows calling fit multiple times
        self.alpha = alpha  # regularization / penalty term (USED IN BONUS)

    def append_ones(self, X):
        # append the 1s columns as feature 0
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def score(self, x):
        if not self.initialized:
            self.w = np.ones(x.shape) / x.shape
        return np.sum(self.w * x, axis=-1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(self.score(x), -25, 25)))

    def compute_gradient(self, x, y):
        gradient = np.zeros((x.shape[1],))

        p = self.sigmoid(x)
        if x.ndim == 1:
            gradient = (p - y) * x
        else:
            e = p - y  
            gradient = (x.T @ e) / x.shape[0]  
        return gradient

    def batch_update(self, batch_X, batch_y):
        g = self.compute_gradient(batch_X, batch_y)
        self.w -= self.eta * g

    def fit(self, X, y, epochs=1, batch_size=1, validation_X=None, validation_y=None):
        if validation_X is None:
            validation_X, validation_y = X, y
        
        metrics = []
        X = self.append_ones(X)

        if not self.initialized:
            self.w = np.ones(X.shape[1]) / X.shape[1]
            self.initialized = True

        for _ in range(epochs):
            self.optimize(X, y, batch_size)
            y_pred = self.predict(validation_X)
            metrics.append(accuracy_score(validation_y, y_pred))

        return np.array(metrics)

    def predict(self, X):
        X_aug = self.append_ones(X)
        p = self.sigmoid(X_aug)
        return (p >= 0.5).astype(int)

    def optimize(self, X, y, batch_size):
        """
        Perform one epoch batch gradient on shuffled data
        :param X: 'np.array' of shape (num_samples, num_features +1), The training data with zero-th column appended
        :param y: target values of shape (num_samples,)
        :param batch_size: batch_size of the batch_update
        :return: None
        """

        indices = np.random.permutation(len(X))  # used to shuffle the data
        for i in range(0, X.shape[0], batch_size):
            batch_x = X[indices[i:i + batch_size]]
            batch_y = y[indices[i:i + batch_size]]
            self.batch_update(batch_x, batch_y)