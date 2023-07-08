import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    print("Theta is: ", logreg.theta)
    print("The model accuracy is: ", logreg.predict(x_eval))


    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def h(theta, x):
            return 1 / (1 + np.exp(-np.dot(x, theta)))
        
        def gradient(theta, x, y):
            m, _ = x.shape()
            return -1 / m * np.dot(x.T, (y - h(theta, x))) 
        
        def hessian(theta, x):
            m, _ = x.shape()
            return 1 / m * np.dot(x.T, h(theta, x)*(1 - h(theta, x))*x)
        
        def next_theta(theta, x, y):
            return theta - np.dot(np.linalg.inv(hessian(theta, x)), gradient(theta, x, y))
        
        m,n = x.shape()

        if self.theta is None:
            self.theta = np.zeros(n)
        
        old_theta = self.theta
        new_theta = next_theta(self.theta, x, y)
        while np.linalg.norm(new_theta-old_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(self.theta, x, y)

        self.theta = new_theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***
