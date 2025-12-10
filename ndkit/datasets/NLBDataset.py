import os
import pickle
import numpy as np
from torch.utils.data import Dataset

from ndkit.datasets.utils import partition, seq_process
from .registry import register_dataset

TRAINVAL_SPLIT_RATIO = 0.9

@register_dataset("NLB")
class NLBDataset(Dataset):
    """
    NLB-style dataset for a single split: 'train' / 'val' / 'test'.

    After processing, attributes:
        self.x : np.ndarray, shape (N, T, C) or (N, C)
        self.y : np.ndarray, shape (N, D) or (N,)
        self.x_dim, self.y_dim, self.seq_len : parsed in `_parse_shapes`
    """

    def __init__(self, cfg, split="train"):
        self.process_data(cfg, split)
        self._parse_shapes()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

    def process_data(self, cfg, split: str):
        """Load, split and preprocess data according to config and split."""
        data_path = os.path.join(cfg.data_dir, f"{cfg.session}.pickle")
        with open(data_path, "rb") as f:
            Data = pickle.load(f)

        # Add acceleration if not present
        if "acc" not in Data.keys():
            Data["acc"] = [np.diff(z, axis=1, prepend=z[:, [0]]) for z in Data["vel"]]

        cond = Data["condition"] if "condition" in Data.keys() else [np.nan] * len(Data["spikes"])
        neural = Data["spikes"]
        behavior = Data[cfg.kin.type]

        # Train / val / test split indices
        train_idx, test_idx = partition(cond, cfg.split.test_frac) # TDDO: fix random seed for fair comparison
        split_point = int(len(train_idx) * TRAINVAL_SPLIT_RATIO)
        val_idx = train_idx[split_point:]
        train_idx = train_idx[:split_point]

        if split == "train":
            neural = [neural[i] for i in train_idx]
            behavior = [behavior[i] for i in train_idx]
        elif split == "val":
            neural = [neural[i] for i in val_idx]
            behavior = [behavior[i] for i in val_idx]
        elif split == "test":
            neural = [neural[i] for i in test_idx]
            behavior = [behavior[i] for i in test_idx]
        else:
            raise ValueError(f"Unsupported split: {split}")
        
        # Sequence preprocessing: binning, lagging etc.
        neural, behavior = seq_process(cfg.neural.Delta, cfg.neural.tau_prime, neural, behavior)

        self.x = neural.astype(np.float32)
        self.y = behavior.astype(np.float32)

    def _parse_shapes(self):
        """Parse x_dim, seq_len, y_dim for later model construction."""
        # x shape
        if self.x.ndim == 3:              # (N, T, C)
            self.seq_len, self.x_dim = self.x.shape[1:]
        elif self.x.ndim == 2:            # (N, C)
            self.x_dim = self.x.shape[1]
            self.seq_len = None
        else:
            raise ValueError(f"Unsupported x shape: {self.x.shape}")

        # y shape
        if self.y.ndim == 1:              # (N,)
            self.y_dim = 1
        elif self.y.ndim == 2:            # (N, D)
            self.y_dim = self.y.shape[1]
        else:
            raise ValueError(f"Unsupported y shape: {self.y.shape}")