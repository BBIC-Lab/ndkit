"""
This implementation is adapted from the MATLAB code released with the paper:

    Qi Yu*, Xinyun Zhu*, Xinzhu Xiong, et al. (2025)
    "Human motor cortex encodes complex handwriting through a sequence of stable neural states."
    Zenodo: https://zenodo.org/records/14865736

The original MATLAB implementation was converted to Python by Jie Yu.
"""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .registry import register_model


# ======================================================================
# Main Model
# ======================================================================

@register_model("DyEnsemble")
class Model:
    """Dynamic Ensemble Encoding Model + Particle Filter Predictor."""

    def __init__(self, cfg):
        self.cfg = cfg

        # System dynamics (learned)
        self.A = None
        self.Q = None

        # Learned encoding models
        self.encoding_models = None

        # Configurable hyperparameters
        self.n_encoding_models = getattr(cfg, "n_encoding_models", 5)
        self.alpha = getattr(cfg, "alpha", 0.9)             # mode prior decay
        self.n_particals = getattr(cfg, "n_particals", 200)

        # Internal state for prediction
        self.pred_dim = None
        self.x = None   # particle states
        self.p = None   # mode probabilities


    # ==================================================================
    # Fit model
    # ==================================================================
    def fit(self, xxx, yyy, train_cfg=None):
        """
        X: behavioral signal, np.ndarray (n_beh, T)
        Y: neural signal,     np.ndarray (n_neur, T)
        """
        # TODO
        X = yyy.T
        Y = xxx.T

        # --------------------------------------------------------------
        # Read training hyperparameters
        # --------------------------------------------------------------
        train_cfg = train_cfg or {}
        tol_err = getattr(train_cfg, "tol_err", 1e5)
        err_threshold = getattr(train_cfg, "err_threshold", 1e-5)
        max_epoch = getattr(train_cfg, "max_epoch", 50)

        # --------------------------------------------------------------
        # Learn latent dynamics: X(t+1) = A X(t), noise ~ Q
        # --------------------------------------------------------------
        A, Q = solve_mat(X[:, :-1], X[:, 1:])
        self.A = A
        self.Q = Q
        self.pred_dim = A.shape[0]

        # --------------------------------------------------------------
        # Initialize encoding models
        # --------------------------------------------------------------
        encoding_models = []
        T = X.shape[1]
        random_indices = torch.randperm(T)

        init_block = min(3000, T)
        block_size = init_block // self.n_encoding_models

        for i in range(self.n_encoding_models):
            mask = torch.arange(Y.shape[0]).long()
            start = i * block_size
            end = start + min(block_size, T)

            model = MaskMLP(X.shape[0], Y.shape[0], mask=mask)

            ds = SimpleDataset(
                X[:, random_indices[start:end]].T,
                Y[:, random_indices[start:end]].T
            )
            train_encoding_model(model, ds)
            encoding_models.append(model)

        # --------------------------------------------------------------
        # TFC (Temporal Functional Clustering)
        # --------------------------------------------------------------
        tfc_dataset = SimpleDataset(X.T, Y.T)
        tfc_bar = tqdm(range(max_epoch), desc="TFC_train", leave=False)

        for epoch in tfc_bar:
            tfc_bar.set_description(f"TFC_epoch:{epoch}")

            # -----------------------------------------
            # E-step: assign each time point to a model
            # -----------------------------------------
            errs = np.zeros((T, self.n_encoding_models))

            X_input = tfc_dataset.x
            for j, model in enumerate(encoding_models):
                preds = model(X_input).detach().numpy()
                y_true = tfc_dataset.y[:, model.mask].numpy()
                errs[:, j] = np.sum((preds - y_true) ** 2, axis=1)

            best_idx = np.argmin(errs, axis=1)

            # -----------------------------------------
            # M-step: retrain each model
            # -----------------------------------------
            for j, model in enumerate(encoding_models):
                jX = X[:, best_idx == j]
                jY = Y[:, best_idx == j]

                if jX.shape[1] > 0:
                    ds = SimpleDataset(jX.T, jY.T)
                    train_encoding_model(model, ds)

            # Early stopping
            new_err = np.min(errs, axis=1).mean()
            change = abs(new_err - tol_err) / max(tol_err, 1e-12)
            tol_err = new_err

            tfc_bar.set_postfix({"tfc_loss": tol_err})

            if change < err_threshold:
                print(f"[TFC] Early stopping at epoch={epoch}, tol_err={tol_err:.6f}")
                break

        self.encoding_models = encoding_models


    # ==================================================================
    # Predict
    # ==================================================================
    def predict(self, Y):
        """
        Y : neural data [n_neur, T]
        Returns: predicted latent state [pred_dim, T]
        """

        T = Y.shape[1]

        # Initialize particles
        self.x = np.zeros((self.pred_dim, self.n_particals))
        self.p = np.ones(self.n_encoding_models) / self.n_encoding_models

        preds = []

        for t in range(T):
            preds.append(self.step(Y[:, t]))

        return np.stack(preds, axis=1)


    # ==================================================================
    # Save and Load
    # ==================================================================
    def save(self, path):
        """
        Save model parameters. Encoding models stored via pickle.
        """
        data = {
            "A": self.A,
            "Q": self.Q,
            "models": self.encoding_models,
            "pred_dim": self.pred_dim,
            "n_parts": self.n_particals,
            "n_models": self.n_encoding_models,
            "alpha": self.alpha,
        }
        np.save(path, data, allow_pickle=True)

    def load(self, path):
        """
        Load model parameters.
        """
        data = np.load(path, allow_pickle=True).item()

        self.A = data["A"]
        self.Q = data["Q"]
        self.encoding_models = data["models"]
        self.pred_dim = data["pred_dim"]

        self.n_particals = data["n_parts"]
        self.n_encoding_models = data["n_models"]
        self.alpha = data["alpha"]


    # ==================================================================
    # Single filtering step
    # ==================================================================
    @torch.no_grad()
    def step(self, z):
        """
        Run particle filter update for a single time step
        """

        rng = np.random.default_rng(seed=42)

        # --------------------------------------------------------------
        # Propagate particles
        # --------------------------------------------------------------
        noise = rng.multivariate_normal(
            mean=np.zeros(self.pred_dim),
            cov=self.Q,
            size=self.n_particals
        ).T

        self.x = self.A @ self.x + noise

        # --------------------------------------------------------------
        # Update mode priors
        # --------------------------------------------------------------
        self.p = (self.p ** self.alpha)
        self.p = self.p / self.p.sum()

        # --------------------------------------------------------------
        # Compute likelihoods from each model
        # --------------------------------------------------------------
        weights = []
        Cs = []

        for model in self.encoding_models:
            w, C = model.get_prob(self.x, z)
            weights.append(w)
            Cs.append(C)

        Cmax = np.max(Cs)
        prob = np.zeros_like(self.p)

        for i in range(self.n_encoding_models):
            prob[i] = weights[i].mean() * np.exp(Cs[i] - Cmax)

        # Avoid numeric collapse
        self.p = self.p * prob
        self.p = np.maximum(self.p, 1e-8)
        self.p = self.p / self.p.sum()

        # --------------------------------------------------------------
        # Combine predicted particles into final estimate
        # --------------------------------------------------------------
        final_w = np.zeros(self.n_particals)
        for i in range(self.n_encoding_models):
            final_w += self.p[i] * weights[i] * np.exp(Cs[i] - Cmax)

        final_w = final_w / final_w.sum()

        pred = (self.x * final_w.reshape(1, -1)).sum(axis=1)

        # Resample particles
        idx = rng.choice(self.n_particals, size=self.n_particals, p=final_w)
        self.x = self.x[:, idx]

        return pred


# ======================================================================
# Helper modules
# ======================================================================

class MaskMLP(nn.Module):
    """MLP that only predicts a masked subset of outputs."""

    def __init__(self, in_channel, out_channel, mask=None):
        super().__init__()

        if mask is None:
            half = out_channel // 2
            mask = torch.randperm(out_channel)[:half]

        self.mask = nn.Parameter(mask, requires_grad=False)

        self.fc = nn.Linear(in_channel, out_channel)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        out = self.fc(x)
        return out[:, self.mask]

    def get_loss(self, x, y):
        return self.loss_fn(self.forward(x), y[:, self.mask])

    def get_prob(self, x, z):
        Sigma = self.Sigma
        preds = self.forward(torch.Tensor(x.T)).numpy().T
        z = z[self.mask]

        diff = preds - z.reshape(-1, 1)
        tmp = np.einsum("in,ij->jn", diff, Sigma)
        w = np.einsum("jn,jn->n", tmp, diff)

        C = (-w / 2).max()
        w = np.exp(-w / 2 - C)

        return w, C

    def after_training(self, dataloader):
        err = []
        with torch.no_grad():
            for x, y in dataloader:
                pred = self.forward(x)
                err.append((pred - y[:, self.mask]).numpy())

        err = np.concatenate(err, axis=0)
        R = np.cov(err.T)
        self.Sigma = np.linalg.inv(R)


def train_encoding_model(model, dataset):
    """Train mask-based encoder."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(15):
        for x, y in loader:
            loss = model.get_loss(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.after_training(loader)


def solve_LS(X, Y):
    theta = (np.linalg.inv(X @ X.T) @ X @ Y).T
    return theta


def solve_mat(X, Y):
    d, _ = Y.shape
    Theta = np.stack([solve_LS(X, Y[i]) for i in range(d)])
    err = Y - Theta @ X
    return Theta, np.cov(err)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
