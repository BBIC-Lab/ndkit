import torch.nn as nn
from omegaconf import ListConfig

class MlpBlock(nn.Module):
    def __init__(self, input_size, output_size=None, hidden_sizes=[], activation='relu', dropout=0):
        super(MlpBlock, self).__init__()
        self.input_size = input_size
        if isinstance(hidden_sizes, (list, ListConfig)): # TODO: better way to handle ListConfig
            hidden_sizes = list(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.activation_type = activation
        self.dropout_rate = dropout
        
        self.mlp = self._build_model()

    def _get_activation_function(self):
        activation_class = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity()
        }
        if self.activation_type in activation_class:
            return activation_class[self.activation_type]
        else:
            raise NotImplementedError(f"Activation function '{self.activation_type}' is not implemented.")

    def _build_model(self):
        num_layers = len(self.hidden_sizes) + (1 if self.output_size is not None else 0)
        assert num_layers > 0, "Must provide at least one layers"
        layer_dims = [self.input_size] + self.hidden_sizes + ([self.output_size] if self.output_size is not None else [])

        layers = []
        for i in range(len(layer_dims) - 2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(self._get_activation_function())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)