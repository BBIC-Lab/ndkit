import torch.nn as nn
import torch
import math

from .registry import register_model

@register_model("Transformer")
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_size = getattr(cfg, "input_size")
        output_size = getattr(cfg, "output_size")
        d_model = getattr(cfg, "d_model")
        num_heads = getattr(cfg, "num_heads")
        d_ff = getattr(cfg, "d_ff")
        num_layers = getattr(cfg, "num_layers")
        dropout = getattr(cfg, "dropout", 0.0)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_projection = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = self.input_projection(x) # (batch_size, seq_len, d_model)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (batch_size, 1 + seq_len, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # (1 + seq_len, batch_size, d_model)
        x = self.transformer_encoder(x) 
        x = x[0, :, :] # (batch_size, d_model)
        x = self.fc(x) # (batch_size, output_size)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * -(math.log(10000.0) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]