import numpy as np
import random
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    - seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode="max", relative=False):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.relative = relative

        self.best_score = np.inf if mode == "min" else -np.inf
        self.best_epoch = -1
        self.counter = 0
        self.early_stop = False

    def __call__(self, epoch, score):
        if self.relative:
            if self.mode == "min":
                improved = score < self.best_score * (1 - self.delta)
            else:
                improved = score > self.best_score * (1 + self.delta)
        else:
            if self.mode == "min":
                improved = score < self.best_score - self.delta
            else:
                improved = score > self.best_score + self.delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return self.early_stop, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop, False

