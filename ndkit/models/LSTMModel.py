import torch.nn as nn
from .registry import register_model


@register_model("LSTM")
class Model(nn.Module):
    """
    A LSTM-based recurrent neural network model for sequence modeling.

    Args:
        cfg: A config object containing model hyperparameters.
    """

    def __init__(self, cfg):
        super().__init__()
        input_size = getattr(cfg, "input_size")
        output_size = getattr(cfg, "output_size")
        hidden_size = getattr(cfg, "hidden_size")
        num_layers = getattr(cfg, "num_layers")
        dropout = getattr(cfg, "dropout", 0.0)
        bidirectional = getattr(cfg, "bidirectional", False)
        self.pooling = getattr(cfg, "pooling", "last")

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * (2 if bidirectional else 1), output_size),
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        enc_out, _ = self.encoder(x)

        if self.pooling == "last":
            enc_out = enc_out[:, -1, :]               # (B, D)
        elif self.pooling == "mean":
            enc_out = enc_out.mean(dim=1)             # (B, D)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        dec_out = self.projection(enc_out)
        return dec_out