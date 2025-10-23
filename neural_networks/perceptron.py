import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, epochs=25, shuffle=True):
        self.lr = lr
        self.epochs = epochs
        self.shuffle = shuffle
        self.w = None
        self.b = 0.0

    def _init_weights(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_epochs):
            idx = np.arange(n_samples)
            if self.shuffle:
                np.random.shuffle(idx)

            for i in idx:
                z = np.dot(self.w, X[i]) + self.b
                y_hat = 1 if z >= 0 else 0
                err = y[i] - y_hat
                if err != 0:
                    self.w += self.lr * err * X[i]
                    self.b += self.lr * err
        return self