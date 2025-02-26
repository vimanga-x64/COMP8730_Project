"""Encoder Architecture"""

import torch
import torch.nn as nn
import math
from typing import Optional
#########################################
# 1. Positional Encoding Module
#########################################
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as in Vaswani et al. (2017).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


#########################################
# 2. Transformer Encoder Module
#########################################
class TransformerCharEncoder(nn.Module):
    """
    Transformer-based character encoder that mirrors Girrbach's BiLSTMEncoder interface.

    Args:
        input_size: Dimensionality of input features.
        embed_dim: Embedding dimension (must be divisible by num_heads).
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        projection_dim: Optional projection dimension for output (if None, returns embed_dim).
        max_len: Maximum sequence length for positional encodings.
    """

    def __init__(
            self,
            input_size: int,
            embed_dim: int,
            num_layers: int = 1,
            num_heads: int = 8,
            dropout: float = 0.1,
            projection_dim: Optional[int] = None,
            max_len: int = 5000,
    ):
        super(TransformerCharEncoder, self).__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.projection_dim = projection_dim

        # Project input features if necessary.
        if input_size != embed_dim:
            self.input_projection = nn.Linear(input_size, embed_dim)
        else:
            self.input_projection = nn.Identity()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)

        # Create Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional projection to a different dimension.
        if projection_dim is not None:
            self.reduce_dim = nn.Linear(embed_dim, projection_dim)
        else:
            self.reduce_dim = nn.Identity()

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch_size, seq_len, input_size)
            lengths: Tensor of shape (batch_size,) containing valid lengths.
        Returns:
            Tensor of shape (batch_size, seq_len, projection_dim) or (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = inputs.size()

        # Project inputs.
        x = self.input_projection(inputs)  # (batch_size, seq_len, embed_dim)

        # Add positional encodings.
        x = self.pos_encoder(x)  # (batch_size, seq_len, embed_dim)

        # Transformer expects (seq_len, batch_size, embed_dim)
        # x = x.transpose(0, 1)

        # Create key padding mask: True for padded positions.
        device = inputs.device
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1)

        # Process using Transformer encoder.
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        # encoded = encoded.transpose(0, 1)  # (batch_size, seq_len, embed_dim)

        # Optionally reduce dimension.
        encoded = self.reduce_dim(encoded)
        return encoded


#########################################
# 3. Sample Script to Test the Encoder
#########################################
def main():
    # Parameters
    batch_size = 2
    seq_len = 10
    input_size = 16  # Dimensionality of input character embeddings
    embed_dim = 32  # Transformer internal embedding size
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    projection_dim = 16  # Optional output projection
    max_len = 100

    # Create a sample input tensor with random values.
    # This could represent, for example, character embeddings for a batch of sequences.
    sample_input = torch.randn(batch_size, seq_len, input_size)

    # Define sequence lengths (e.g., first sequence has length 10, second sequence length 8).
    lengths = torch.tensor([10, 8])

    # Instantiate the TransformerCharEncoder.
    encoder = TransformerCharEncoder(
        input_size=input_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        projection_dim=projection_dim,
        max_len=max_len
    )

    # Pass the sample input through the encoder.
    output = encoder(sample_input, lengths)

    # Print output details.
    print("Output shape:", output.shape)
    print("Output tensor:", output)


if __name__ == '__main__':
    main()