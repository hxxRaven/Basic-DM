import numpy as np

class NearsNeighbor:
    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Y_pred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distences = np.sum(np.abs(self.Xtr - X[i, : ]), axis=1)
            min_index = np.argmin(distences)
            Y_pred[i] = self.ytr[min_index]

        return Y_pred
