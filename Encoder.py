"""Encoder Architecture"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in
    Vaswani et al. (2017). This injects sequence order information
    into the embeddings since the Transformer is inherently permutation-invariant.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        # Register pe as a buffer so it is saved in the model state but not updated by gradients.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        Returns: Tensor of the same shape with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerCharEncoder(nn.Module):
    """
    A character-based Transformer encoder that maps an input sequence of character indices
    into a sequence of contextualized representations.

    Parameters:
    - vocab_size: Size of the character vocabulary.
    - d_model: Dimensionality of embeddings and encoder hidden states.
    - nhead: Number of heads in multi-head attention.
    - num_layers: Number of Transformer encoder layers.
    - dim_feedforward: Dimensionality of the feed-forward network within each Transformer layer.
    - dropout: Dropout probability.
    - max_len: Maximum sequence length supported.
    """

    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=512, dropout=0.1, max_len=512):
        super(TransformerCharEncoder, self).__init__()
        self.d_model = d_model

        # Embedding layer for characters
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding to incorporate token order information
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # Transformer encoder: stack of encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: Tensor of shape (batch_size, seq_len) containing character indices.
        src_mask: Optional mask to prevent attention to certain positions.
        src_key_padding_mask: Optional mask for padded positions.

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model) with contextualized representations.
        """
        # Embed the input and scale embeddings by sqrt(d_model)
        x = self.embedding(src) * math.sqrt(self.d_model)  # Shape: (batch_size, seq_len, d_model)

        # Add positional encodings
        x = self.pos_encoder(x)

        # Transformer expects input shape (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)

        # Pass through Transformer encoder layers
        encoded = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Transpose back to (batch_size, seq_len, d_model)
        output = encoded.transpose(0, 1)
        return output


# Example usage:
if __name__ == "__main__":
    # Suppose our character vocabulary size is 100 (including special characters)
    vocab_size = 100
    batch_size = 32
    seq_len = 50
    d_model = 256

    # Create the encoder instance
    encoder = TransformerCharEncoder(vocab_size, d_model=d_model, nhead=8, num_layers=6,
                                     dim_feedforward=512, dropout=0.1, max_len=512)

    # Dummy input: random character indices with shape (batch_size, seq_len)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Obtain the contextualized representations
    encoded_output = encoder(dummy_input)
    print("Encoded output shape:", encoded_output.shape)
    # Expected output shape: (batch_size, seq_len, d_model)