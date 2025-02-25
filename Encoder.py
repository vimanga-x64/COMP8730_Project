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
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
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
        x = x.transpose(0, 1)

        # Create key padding mask: True for padded positions.
        device = inputs.device
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1)

        # Process using Transformer encoder.
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        encoded = encoded.transpose(0, 1)  # (batch_size, seq_len, embed_dim)

        # Optionally reduce dimension.
        encoded = self.reduce_dim(encoded)
        return encoded

# Example usage:
if __name__ == "__main__":
    # Set parameters
    vocab_size = 100  # For example, 100 characters in vocabulary.
    batch_size = 32
    seq_len = 50  # Each sentence is 50 characters long.
    d_model = 256  # Model dimensionality.
    nhead = 8
    num_layers = 6
    dim_feedforward = 512
    dropout = 0.1
    max_len = 512

    # Instantiate the Transformer-based encoder.
    encoder = TransformerCharEncoder(
    )

    # Create a dummy input tensor (batch, seq_len) with random indices in the range [0, vocab_size).
    dummy_input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long)
    # For this example, assume that all sequences have full length.
    sentence_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

    # Compute embeddings (or let the encoder do it internally).
    # If your encoder expects raw indices, call it with dummy_input; if it expects embeddings,
    # then first compute the embeddings.
    # In our updated implementation, we want to pass embeddings and sentence_lengths.
    # So we compute the embeddings using the encoder's own embedding layer.
    embeddings = encoder.embedding(dummy_input) * math.sqrt(d_model)

    # Now run the encoder.
    encodings = encoder(embeddings, sentence_lengths)

    # Print the output shape.
    print("Input shape:", dummy_input.shape)
    print("Encoded output shape:", encodings.shape)
    print("Expected shape: ", (batch_size, seq_len, d_model))