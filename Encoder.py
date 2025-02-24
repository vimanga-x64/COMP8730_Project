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
        pe[:, 0::2] = torch.sin(position * div_term)  # sin for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos for odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
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
    A character-based Transformer encoder that maps a sequence of character embeddings
    into contextualized representations.

    Parameters:
    - vocab_size: Size of the character vocabulary (used for embedding lookup when needed).
    - d_model: Dimensionality of embeddings and encoder hidden states.
    - nhead: Number of heads in multi-head attention.
    - num_layers: Number of Transformer encoder layers.
    - dim_feedforward: Dimensionality of the feed-forward network.
    - dropout: Dropout probability.
    - max_len: Maximum sequence length supported.

    Now, the forward method expects:
       embeddings: Tensor of shape (batch, seq_len, d_model)
           (precomputed via self.embedding(sentences))
       sentence_lengths: Tensor of shape (batch,) indicating valid lengths.
    """

    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=512, dropout=0.1, max_len=512):
        super(TransformerCharEncoder, self).__init__()
        self.d_model = d_model

        # Embedding layer (if needed)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding module
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # Transformer encoder layer with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, embeddings, sentence_lengths, src_mask=None):
        """
        embeddings: Tensor of shape (batch, seq_len, d_model) (precomputed embeddings)
        sentence_lengths: Tensor of shape (batch,) with valid lengths.
        src_mask: Optional mask for attention.

        Returns:
            output: Tensor of shape (batch, seq_len, d_model) with contextualized representations.
        """
        batch_size, seq_len, _ = embeddings.size()
        device = embeddings.device

        # Create a key padding mask: True for positions >= sentence_lengths (i.e., padding positions)
        idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        src_key_padding_mask = idx >= sentence_lengths.unsqueeze(1)

        # Apply positional encoding to the embeddings.
        x = self.pos_encoder(embeddings)

        # Pass through Transformer encoder. Note: batch_first=True means x remains (batch, seq_len, d_model)
        encoded = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return encoded


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