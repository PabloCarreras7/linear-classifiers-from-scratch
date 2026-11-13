"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
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
        self.x_mean = None
        self.x_std = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Standardizes input using training statistics."""
        if self.x_mean is None or self.x_std is None:
            return X
        return (X - self.x_mean) / (self.x_std + 1e-8)

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        N, D = X_train.shape

        # Compute class scores for all examples at once
        scores = X_train @ self.w.T  # (N, C)

        # Get score of the correct class for each example
        true_scores = scores[np.arange(N), y_train].reshape(-1, 1)

        # Compute hinge margins
        margins = np.maximum(0, scores - true_scores + 1)
        margins[np.arange(N), y_train] = 0

        # Build gradient mask
        mask = (margins > 0).astype(float)

        # Correct class gets negative sum of violations
        violation_counts = np.sum(mask, axis=1)
        mask[np.arange(N), y_train] = -violation_counts

        # Average gradient with L2 regularization
        dW = mask.T @ X_train / N + self.reg_const * self.w

        return dW

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape

        # Compute and store training statistics for standardization
        self.x_mean = np.mean(X_train, axis=0)
        self.x_std = np.std(X_train, axis=0)
        X_norm = self._standardize(X_train)

        np.random.seed(42)
        self.w = np.random.uniform(-1, 1, (self.n_class, D)) * 0.01

        batch_size = 200
        active_lr = self.lr

        for epoch in range(self.epochs):
            # Halve the learning rate every 20 epochs
            if epoch > 0 and epoch % 20 == 0:
                active_lr *= 0.5

            idx = np.random.permutation(N)
            X_shuffled = X_norm[idx]
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
        # Standardize test data using training statistics
        X_norm = self._standardize(X_test)
        scores = X_norm @ self.w.T
        return np.argmax(scores, axis=1)