import torch.nn as nn
from .registry import register_model
from .layers.mlp import MlpBlock

@register_model("FFN") # or "DNN"
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_size = getattr(cfg, "input_size")
        output_size = getattr(cfg, "output_size")
        hidden_sizes = getattr(cfg, "hidden_sizes")
        activation = getattr(cfg, "activation", "relu")
        dropout = getattr(cfg, "dropout", 0.0)

        self.mlp = MlpBlock(input_size=input_size, 
                            output_size=output_size, 
                            hidden_sizes=hidden_sizes,
                            activation=activation,
                            dropout=dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input if necessary
        return self.mlp(x)