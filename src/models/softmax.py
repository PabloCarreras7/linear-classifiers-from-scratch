"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        N, D = X_train.shape

        # Compute raw class scores for each example
        logits = X_train @ self.w.T  # (N, C)

        # Numerical stability trick: subtract row-wise maximum
        logits -= np.max(logits, axis=1, keepdims=True)

        # Convert scores to probabilities using softmax
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Subtract 1 from correct class probabilities to get gradient signal
        probabilities[np.arange(N), y_train] -= 1

        # Average over batch and add L2 regularization term
        dW = probabilities.T @ X_train / N + self.reg_const * self.w

        return dW

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape

        np.random.seed(42)
        self.w = np.random.uniform(-1, 1, (self.n_class, D)) * 0.01

        batch_size = 200
        active_lr = self.lr

        for epoch in range(self.epochs):
            # Exponential learning rate decay: shrink lr by 5% each epoch
            active_lr = self.lr * (0.95 ** epoch)

            idx = np.random.permutation(N)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                dW = self.calc_gradient(X_batch, y_batch)
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
        # Compute scores and return index of highest scoring class
        logits = X_test @ self.w.T
        return np.argmax(logits, axis=1)