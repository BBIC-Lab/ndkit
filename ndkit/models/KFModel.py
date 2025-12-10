import numpy as np
import pickle

from .registry import register_model


@register_model("KF")
class Model:
    """
    Kalman Filter Regression Decoder
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Matrices
        self.A = None   # State transition
        self.H = None   # Observation matrix
        self.W = None   # Process noise
        self.Q = None   # Observation noise

        # KF internal variables
        self.P = None
        self.I = None
        self.dim_x = 0

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------
    def fit(self, x, y, train_cfg=None):
        """
        Fit the Kalman Filter model.

        Parameters
        ----------
        x: np.array [T, N_neurons]
        y: np.array [T, N_outputs]
        """

        X = y.T           # shape: [pos_dim, T]
        X1 = X[:, :-1]    # X(t)
        X2 = X[:, 1:]     # X(t+1)
        Z = x.T           # spikes, shape [neuron_dim, T]

        pos_dim, T = X.shape

        # State transition matrix A
        A = (X2 @ X1.T) @ np.linalg.pinv(X1 @ X1.T)

        # Observation matrix H
        H = (Z @ X.T) @ np.linalg.pinv(X @ X.T)

        # Process noise
        W = (X2 - A @ X1) @ (X2 - A @ X1).T / (T - 1)

        # Observation noise
        Q = (Z - H @ X) @ (Z - H @ X).T / T

        self.A = A
        self.H = H
        self.W = W
        self.Q = Q

        # KF dimensions
        self.dim_x = pos_dim
        self.P = np.eye(pos_dim)
        self.I = np.eye(pos_dim)

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------
    def predict(self, x):
        """
        Run Kalman filtering (prediction only, no parameter update).

        x: np.array [T, neuron_dim]
        Returns: np.array [T, pos_dim]
        """

        if any(v is None for v in [self.A, self.H, self.W, self.Q]):
            raise RuntimeError("Model parameters are not fitted. Call fit() first.")

        T = x.shape[0]
        P = self.P.copy()

        # Initialize state trajectory
        x_pre = np.zeros((self.dim_x, T))
        x_pre[:, 0] = np.zeros(self.dim_x)

        for t in range(1, T):
            # Prediction
            X_ = self.A @ x_pre[:, t - 1]
            P_ = self.A @ P @ self.A.T + self.W

            # Kalman gain
            K = (P @ self.H.T) @ np.linalg.pinv(self.H @ P_ @ self.H.T + self.Q)

            # Correction
            x_pre[:, t] = X_ + K @ (x[t, :].T - self.H @ X_)
            P = (self.I - K @ self.H) @ P_

        return x_pre.T

    # ---------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------
    def save(self, path):
        """Serialize model parameters into a .pkl file."""
        state = {
            "A": self.A,
            "H": self.H,
            "W": self.W,
            "Q": self.Q,
            "P": self.P,
            "I": self.I,
            "dim_x": self.dim_x,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path):
        """Load model parameters from a .pkl file."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.A = state["A"]
        self.H = state["H"]
        self.W = state["W"]
        self.Q = state["Q"]
        self.P = state["P"]
        self.I = state["I"]
        self.dim_x = state["dim_x"]