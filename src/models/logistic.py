"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.mean = None
        self.std = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # Compute sigmoid separately for positive and negative values
        # to avoid numerical overflow
        result = np.zeros_like(z, dtype=float)
        pos = z >= 0
        result[pos] = 1 / (1 + np.exp(-z[pos]))
        result[~pos] = np.exp(z[~pos]) / (1 + np.exp(z[~pos]))
        return result

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape

        # Compute and store training statistics for normalization
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)
        X_train = (X_train - self.mean) / (self.std + 1e-8)

        # Add bias term
        X_train = np.hstack([X_train, np.ones((N, 1))])
        N, D = X_train.shape

        # Initialize weights with small random values
        np.random.seed(42)
        self.w = np.random.uniform(-1, 1, D) * 0.01

        for epoch in range(self.epochs):
            # Exponential learning rate decay
            active_lr = self.lr * (0.95 ** epoch)

            # Compute predictions
            scores = X_train @ self.w
            predictions = self.sigmoid(scores)

            # Gradient of binary cross entropy loss
            error = predictions - y_train
            dW = X_train.T @ error / N

            # Update weights
            self.w -= active_lr * dW

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
        # Normalize using training statistics
        X_test = (X_test - self.mean) / (self.std + 1e-8)

        # Add bias term
        N = X_test.shape[0]
        X_test = np.hstack([X_test, np.ones((N, 1))])

        # Apply sigmoid and threshold to get class predictions
        probabilities = self.sigmoid(X_test @ self.w)
        return (probabilities >= self.threshold).astype(int)