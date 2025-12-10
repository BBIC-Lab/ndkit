import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted
from .registry import register_model

@register_model("iTransformer")
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, cfg):
        super().__init__()
        seq_len = getattr(cfg, "seq_len")
        input_size = getattr(cfg, "input_size")
        output_size = getattr(cfg, "output_size")
        d_model = getattr(cfg, "d_model")
        n_heads = getattr(cfg, "n_heads")
        d_ff = getattr(cfg, "d_ff")
        e_layers = getattr(cfg, "e_layers")
        dropout = getattr(cfg, "dropout", 0.0)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
                                                    
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, None, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * input_size, output_size)

    def forward(self, x):
        # Embedding
        enc_out = self.enc_embedding(x, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  
        return output