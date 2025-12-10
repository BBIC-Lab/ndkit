import pickle
from sklearn import linear_model

from .registry import register_model

@register_model("WF")
class Model:
    """
    WienerFilter
    """
    def __init__(self, cfg):
        self.model = None

    def fit(self, x, y, train_cfg):
        """
        x: numpy array [n_samples, n_features]
        y: numpy array [n_samples, n_outputs]
        """
        self.model = linear_model.LinearRegression()
        self.model.fit(x, y)

    def predict(self, x):
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() before predict().")
        return self.model.predict(x)

    def save(self, path):
        """Save the trained model to a .pkl file."""
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")

        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        """Load a model from a .pkl file."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)