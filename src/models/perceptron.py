"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        N, D = X_train.shape
        
        # Add bias term
        X_train = np.hstack([X_train, np.ones((N, 1))])
        N, D = X_train.shape

        np.random.seed(42)
        self.w = np.random.uniform(-1, 1, (self.n_class, D)) * 0.01
        
        # Accumulated weights for averaging
        w_sum = np.zeros_like(self.w)
        count = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(N)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            for i in range(N):
                x = X_train[i]
                y = y_train[i]

                scores = self.w @ x
                y_pred = np.argmax(scores)

                if y_pred != y:
                    self.w[y] += self.lr * x
                    self.w[y_pred] -= self.lr * x
                
                # Accumulate weights
                w_sum += self.w
                count += 1
            
            self.lr *= 0.95
        
        # Use averaged weights
        self.w = w_sum / count
            
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Compute scores for all examples at once and return the predicted class
        N = X_test.shape[0]
        X_test = np.hstack([X_test, np.ones((N, 1))])
        scores = X_test @ self.w.T
        return np.argmax(scores, axis=1)