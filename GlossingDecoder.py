"""Decoder Model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from Encoder import PositionalEncoding

#########################################
# 5. Glossing Decoder with Cross-Attention
#########################################
class GlossingDecoder(nn.Module):
    """
    Transformer decoder that generates gloss tokens using cross-attention over segment representations.
    Now configured to use batch_first=True.
    """
    def __init__(self, gloss_vocab_size: int, embed_dim: int, num_heads: int,
                 ff_dim: int, num_layers: int, dropout: float = 0.1, max_len: int = 5000):
        super(GlossingDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(gloss_vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len)
        # Set batch_first=True in the decoder layer.
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                    dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
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
        # With batch_first=True, no need to transpose.
        decoded = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        logits = self.fc_out(decoded)
        return logits


def main():
    # Dummy parameters.
    batch_size = 2
    tgt_seq_len = 5  # length of target sequence (e.g., gloss tokens)
    mem_len = 7  # length of memory (aggregated segment representations)
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_layers = 2
    dropout = 0.1
    gloss_vocab_size = 10  # dummy vocabulary size
    max_len = 50  # maximum length for positional encoding

    # Instantiate the GlossingDecoder with batch_first=True.
    decoder = GlossingDecoder(
        gloss_vocab_size=gloss_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_len=max_len
    )

    # Create dummy target token indices.
    # For example, random integers in the range [0, gloss_vocab_size-1].
    tgt = torch.randint(0, gloss_vocab_size, (batch_size, tgt_seq_len))

    # Create dummy memory tensor (e.g., aggregated segment representations).
    # Shape: (batch_size, mem_len, embed_dim)
    memory = torch.randn(batch_size, mem_len, embed_dim)

    # Optionally, create a dummy target mask or memory padding mask if needed.
    # For simplicity, we'll let them be None.

    # Forward pass through the decoder.
    logits = decoder(tgt, memory)

    # logits will have shape: (batch_size, tgt_seq_len, gloss_vocab_size)
    print("Dummy target indices:")
    print(tgt)
    print("\nOutput logits shape:", logits.shape)
    print("\nOutput logits:")
    print(logits)


if __name__ == "__main__":
    main()