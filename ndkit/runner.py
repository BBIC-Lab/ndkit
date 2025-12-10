import numpy as np
import os
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ndkit.datasets.get_dataset import get_dataset
from ndkit.utils.metrics import compute_metrics
from ndkit.utils.misc import get_time_dir, format_metrics, str2hash
from ndkit.utils.loggers import CSVLogger, set_logger
from ndkit.utils.training import set_seed, EarlyStopping, count_parameters
from ndkit.models.get_model import get_model


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = None
        self.main_logger = None
        self.result_dir = None
        self.checkpoint_path = None

    # ================================================================
    # Training
    # ================================================================
    def train(self):
        """Main training loop."""
        self.set_result_dir()
        self.main_logger = set_logger(self.result_dir, log_name="train")
        self.csv_logger = CSVLogger(os.path.join(self.result_dir, "csv_results.csv"))
        self.checkpoint_path = os.path.join(self.result_dir, "best_model.pt")
        OmegaConf.save(self.cfg, os.path.join(self.result_dir, "cfg.yaml"))

        self.main_logger.info(
            f"========== Training {self.cfg.model.name} on "
            f"{self.cfg.data.name}, {self.cfg.data.session} =========="
        )
        self.main_logger.info(self.cfg)

        self.set_device()
        self.main_logger.info(f"Training on device: {self.device}")
        set_seed(self.cfg.train.seed)
        self.set_dataset()
        self.set_model()
        self.set_criterion()
        self.set_optimizer()
        self.set_scheduler()

        self.early_stopping = EarlyStopping(
            patience=self.cfg.train.early_stop.patience,
            delta=self.cfg.train.early_stop.delta,
            mode=self.cfg.train.early_stop.mode,
            relative=self.cfg.train.early_stop.relative,
        )

        total_params = count_parameters(self.model)
        self.main_logger.info(
            f"Model has {total_params:,} trainable parameters\n{self.model}"
        )
        self.main_logger.info(
            f"Input size: {self.cfg.model.input_size}, output size: {self.cfg.model.output_size}, seq_len: {self.cfg.model.seq_len}"
        )

        # Epoch loop
        for epoch in range(1, self.cfg.train.n_epochs + 1):
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            self.main_logger.info(f"Epoch {epoch:03d}, train {format_metrics(train_metrics)}")
            self.main_logger.info(f"\tval {format_metrics(val_metrics)}")

            self.csv_logger.log({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            val_score = val_metrics[self.cfg.train.early_stop.metric]
            stop_flag, is_best = self.early_stopping(epoch, val_score)

            if is_best:
                self.save_checkpoint()
                self.main_logger.info(
                    f"Best model updated at epoch {epoch}, score={val_score:.4f}"
                )

            if stop_flag:
                self.main_logger.info(
                    f"Early stopping at epoch {epoch}, best score={self.early_stopping.best_score:.4f}"
                )
                break

        self.main_logger.info("========== Training complete ==========")  
        self.main_logger.info(f"Model saved at: {self.checkpoint_path}")
        self.main_logger.info(f"Config saved at: {os.path.join(self.result_dir, 'cfg.yaml')}")
        return self.early_stopping.best_score

    # ================================================================
    # One epoch training
    # ================================================================
    def train_one_epoch(self):
        """Single training epoch."""
        self.model.train()
        total_loss, outputs, targets = [], [], []

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Forward and loss (supports auxiliary losses)
            model_out = self.model(data)
            output, loss = self._compute_loss_and_output(model_out, target)

            loss.backward()
            self.optimizer.step()

            outputs.append(output.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            total_loss.append(loss.item())

        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        metrics = compute_metrics(outputs, targets)
        return {"loss": np.mean(total_loss), **metrics}

    # ================================================================
    # Validation / test (shared)
    # ================================================================
    def validate(self, data_loader=None, return_outputs=False):
        """Validation or test evaluation."""
        data_loader = data_loader or self.val_loader

        self.model.eval()
        total_loss, outputs, targets = [], [], []

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                model_out = self.model(data)

                output, loss = self._compute_loss_and_output(model_out, target)

                outputs.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                total_loss.append(loss.item())

        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        metrics = compute_metrics(outputs, targets)

        result = {"loss": np.mean(total_loss), **metrics}
        return (result, {"outputs": outputs, "targets": targets}) if return_outputs else result

    # ================================================================
    # Loss handler (handles auxiliary loss)
    # ================================================================
    def _compute_loss_and_output(self, model_out, target):
        """Supports models that return (output, aux_loss)."""
        if isinstance(model_out, (tuple, list)) and len(model_out) == 2:
            output, aux_loss = model_out
            loss = self.criterion(output, target) + aux_loss
        else:
            output = model_out
            loss = self.criterion(output, target)
        return output, loss

    # ================================================================
    # Standalone evaluation API
    # ================================================================
    def eval(self, ckpt_path=None, save_outputs=True):
        """Evaluate the model on the test set."""

        if ckpt_path is not None:
            # Use directory of checkpoint
            self.result_dir = os.path.dirname(ckpt_path)
        else:
            # Fallback: use training result_dir
            if self.result_dir is None:
                raise ValueError(
                    "result_dir is not set. Please pass ckpt_path explicitly when evaluating."
                )
            ckpt_path = os.path.join(self.result_dir, "best_model.pt")

        self.main_logger = set_logger(self.result_dir, log_name="eval")

        if self.device is None:
            self.set_device()
        self.main_logger.info(f"Eval on device: {self.device}")

        if not hasattr(self, "test_loader"):
            self.set_dataset()

        if not hasattr(self, "model"):
            self.set_model()

        if not hasattr(self, "criterion"):
            self.set_criterion()

        ckpt_path = ckpt_path or os.path.join(self.result_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])

        test_metrics, out = self.validate(self.test_loader, return_outputs=True)
        self.main_logger.info(format_metrics(test_metrics))

        if save_outputs:
            np.savez(os.path.join(self.result_dir, "test_outputs_targets.npz"), **out)

        return test_metrics

    # ================================================================
    # Setup methods
    # ================================================================
    def set_dataset(self):
        """Load dataset and construct loaders."""
        train_set, val_set, test_set = get_dataset(self.cfg.data)

        self.train_loader = DataLoader(train_set, batch_size=self.cfg.train.bs, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.cfg.train.bs, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=self.cfg.train.bs, shuffle=False)

        if self.cfg.model.name == "FFN":
            self.cfg.model.input_size = train_set.x_dim * train_set.seq_len
        else:
            self.cfg.model.input_size = train_set.x_dim
        self.cfg.model.output_size = train_set.y_dim
        self.cfg.model.seq_len = train_set.seq_len

    def set_model(self):
        self.model = get_model(self.cfg.model).to(self.device)

    def set_criterion(self):
        if self.cfg.train.loss_name == "MSE":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Unknown loss: {self.cfg.train.loss_name}")

    def set_optimizer(self):
        if self.cfg.train.opt_name == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay
            )
        elif self.cfg.train.opt_name == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay
            )
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.cfg.train.opt_name}")

    def set_scheduler(self):
        self.scheduler = None  # optional

    def set_device(self):
        gpu = getattr(self.cfg.train, "gpu_to_use", 0)
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

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

    # ================================================================
    # Checkpoint
    # ================================================================
    def save_checkpoint(self):
        ckpt = {
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }
        torch.save(ckpt, self.checkpoint_path)
