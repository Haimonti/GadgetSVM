from sklearn.svm import LinearSVC
import numpy as np
from p2pfl.learning.frameworks.base_model import BaseModel

class SklearnSVM(BaseModel):
    """
    Simple wrapper around scikit-learn's LinearSVC for use with p2pfl.
    """

    def __init__(self, C=1.0, max_iter=1000):
        super().__init__()
        self.model = LinearSVC(C=C, max_iter=max_iter)
        self.trained = False

    def train(self, x_train, y_train, *args, **kwargs):
        self.model.fit(x_train, y_train)
        self.trained = True

    def evaluate(self, x_test, y_test, *args, **kwargs):
        acc = self.model.score(x_test, y_test)
        return {"accuracy": acc}

    def predict(self, x):
        if not self.trained:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(x)

    def get_weights(self):
        return {"coef": self.model.coef_, "intercept": self.model.intercept_}

    def set_weights(self, weights):
        self.model.coef_ = weights["coef"]
        self.model.intercept_ = weights["intercept"]
        self.trained = True

def model_build_fn(*args, **kwargs):
    return SklearnSVM(C=1.0, max_iter=1000)
