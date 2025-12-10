import numpy as np
import os
from omegaconf import OmegaConf
from ndkit.datasets.get_dataset import get_dataset
from ndkit.utils.metrics import compute_metrics
from ndkit.utils.misc import get_time_dir, format_metrics, str2hash
from ndkit.utils.loggers import set_logger
from ndkit.utils.training import set_seed
from ndkit.models.get_model import get_model


class FitRunner:
    """
    Runner for models with an internal `fit(X, Y)` training procedure.

    This runner does NOT manage epochs, batches, or optimizer loops.
    It simply extracts numpy training/testing arrays from the dataset
    and calls model.fit() / model.predict().
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.main_logger = None
        self.result_dir = None
        self.checkpoint_path = None

    # ============================================================
    # Training, but without epochs/batches
    # ============================================================
    def train(self):
        self.set_result_dir()
        self.main_logger = set_logger(self.result_dir, log_name="train")
        OmegaConf.save(self.cfg, os.path.join(self.result_dir, "cfg.yaml"))

        self.main_logger.info(
            f"=== Fitting {self.cfg.model.name} on "
            f"{self.cfg.data.name}, {self.cfg.data.session} ==="
        )
        self.main_logger.info(self.cfg)

        set_seed(self.cfg.train.seed)
        self.set_dataset()
        self.set_model()

        self.main_logger.info(
            f"Input size: {self.cfg.model.input_size}, output size: {self.cfg.model.output_size}, seq_len: {self.cfg.model.seq_len}"
        )
        self.main_logger.info(
            f"Training samples: {self.train_x.shape[0]}, Testing samples: {self.test_x.shape[0]}"
        )

        # Fit model
        self.model.fit(self.train_x, self.train_y, self.cfg.train)

        self.main_logger.info("========== Training complete ==========")  
        if hasattr(self.model, "save"): # Save checkpoint (model must implement save() if needed)
            self.checkpoint_path = os.path.join(self.result_dir, "best_model.pkl")
            self.model.save(self.checkpoint_path)
            self.main_logger.info(f"Model saved to {self.checkpoint_path}")
        self.main_logger.info(f"Config saved at: {os.path.join(self.result_dir, 'cfg.yaml')}")

    # ============================================================
    # Evaluation
    # ============================================================
    def eval(self, ckpt_path=None, save_outputs=True):
        if ckpt_path:
            self.result_dir = os.path.dirname(ckpt_path)
        elif self.result_dir is None:
            raise ValueError("result_dir not set. Pass ckpt_path explicitly.")
        else:
            ckpt_path = os.path.join(self.result_dir, "best_model.pkl")

        self.main_logger = set_logger(self.result_dir, log_name="eval")

        if not hasattr(self, "test_x") or not hasattr(self, "test_y"):
            self.set_dataset()

        if not hasattr(self, "model"):
            self.set_model()

        # Load checkpoint if applicable
        if ckpt_path and hasattr(self.model, "load"):
            self.model.load(ckpt_path)

        Y_pred = self.model.predict(self.test_x)
        test_metrics = compute_metrics(Y_pred, self.test_y)

        self.main_logger.info(format_metrics(test_metrics))

        if save_outputs:
            np.savez(os.path.join(self.result_dir, "test_outputs_targets.npz"),
                     outputs=Y_pred, targets=self.test_y)

        return test_metrics

    # ============================================================
    # Dataset & Model Setup
    # ============================================================
    def set_dataset(self):
        train_set, val_set, test_set = get_dataset(self.cfg.data)

        self.train_x = train_set.x
        self.train_y = train_set.y
        self.val_x = val_set.x # not used in FitRunner, but kept for consistency
        self.val_y = val_set.y
        self.test_x = test_set.x
        self.test_y = test_set.y
        
        flatten_input = getattr(self.cfg.model, "flatten_input", False)
        if flatten_input:
            self.cfg.model.input_size = train_set.x_dim * train_set.seq_len
            self.train_x = self.train_x.reshape(self.train_x.shape[0], -1)
            self.val_x = self.val_x.reshape(self.val_x.shape[0], -1)
            self.test_x = self.test_x.reshape(self.test_x.shape[0], -1)
        else:
            self.cfg.model.input_size = train_set.x_dim
        self.cfg.model.output_size = train_set.y_dim
        self.cfg.model.seq_len = train_set.seq_len

    def set_model(self):
        self.model = get_model(self.cfg.model)

    # ============================================================
    # Result Directory
    # ============================================================
    def set_result_dir(self):
        """Create a unique experiment directory."""
        save_dir = "_".join(
            [str2hash(str(self.cfg)), get_time_dir()] +
            ([self.cfg.note] if getattr(self.cfg, "note", None) else [])
        )

        self.result_dir = os.path.join(
            self.cfg.result_root,
            f"{self.cfg.data.name}-{self.cfg.model.name}-{self.cfg.data.session}",
            save_dir,
        )
        os.makedirs(self.result_dir, exist_ok=True)