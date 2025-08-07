from sklearn.linear_model import SGDClassifier
import numpy as np
from p2pfl.learning.frameworks.base_model import BaseModel

class SklearnSGDSVM(BaseModel):
    """
    Approximates Linear SVM using SGDClassifier with hinge loss.
    This version supports incremental training and peer-to-peer weight sharing for p2pfl.
    """

    def __init__(self, alpha=0.0001, max_iter=1, learning_rate='optimal'):
        super().__init__()
        self.model = SGDClassifier(
            loss='hinge',                # Linear SVM loss
            alpha=alpha,                 # Regularization strength
            max_iter=max_iter,           # One iteration per call
            learning_rate=learning_rate,
            warm_start=True,             # Retain weights across rounds
            tol=None                     # Disable early stopping
        )
        self.trained = False

    def train(self, x_train, y_train, *args, **kwargs):
        # First time we train, we must supply classes
        if not self.trained:
            self.model.partial_fit(x_train, y_train, classes=np.array([0, 1]))  # binary classification
        else:
            self.model.partial_fit(x_train, y_train)
        self.trained = True

    def evaluate(self, x_test, y_test, *args, **kwargs):
        acc = self.model.score(x_test, y_test)
        return {"accuracy": acc}

    def predict(self, x):
        if not self.trained:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(x)

    def get_weights(self):
        if not self.trained:
            raise ValueError("Model weights are not available. Train the model first.")
        return {
            "coef": self.model.coef_.copy(),
            "intercept": self.model.intercept_.copy()
        }

    def set_weights(self, weights):
        self.model.coef_ = weights["coef"]
        self.model.intercept_ = weights["intercept"]
        self.model.classes_ = np.array([0, 1])  # Set classes manually for safety
        self.trained = True

def model_build_fn(*args, **kwargs):
    return SklearnSGDSVM(alpha=0.0001, max_iter=1)
