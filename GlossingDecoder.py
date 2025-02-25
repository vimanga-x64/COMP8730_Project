"""Decoder/Glossing Model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from Encoder import PositionalEncoding
#########################################
# 5. Glossing Decoder with Cross-Attention
#########################################
class GlossingDecoder(nn.Module):
    """
    Transformer decoder that generates gloss tokens using cross-attention over segment representations.
    """

    def __init__(self, gloss_vocab_size: int, embed_dim: int, num_heads: int,
                 ff_dim: int, num_layers: int, dropout: float = 0.1, max_len: int = 5000):
        super(GlossingDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(gloss_vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, gloss_vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: Target gloss token indices (batch_size, tgt_seq_len)
            memory: Aggregated segment representations (batch_size, mem_len, embed_dim)
            tgt_mask: Optional mask for target sequence.
            memory_key_padding_mask: Optional mask for memory.
        Returns:
            logits: Tensor (batch_size, tgt_seq_len, gloss_vocab_size)
        """
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.embed_dim)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        tgt_embedded = tgt_embedded.transpose(0, 1)  # (tgt_seq_len, batch_size, embed_dim)
        memory = memory.transpose(0, 1)  # (mem_len, batch_size, embed_dim)
        decoded = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        decoded = decoded.transpose(0, 1)
        logits = self.fc_out(decoded)
        return logits
