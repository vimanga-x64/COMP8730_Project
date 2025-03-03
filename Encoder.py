"""Encoder Architecture with Relative Positional Encoding"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

#########################################
# 1. Positional Encoding Module (Absolute)
#########################################
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as in Vaswani et al. (2017).
    This module is kept for backward compatibility.
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
# 2. Relative Multihead Attention Module
#########################################
class RelativeMultiheadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding using a learned bias.
    This implementation follows the relative attention bias approach
    similar to that in T5 (Raffel et al., 2020) where a bias is added based
    on the relative distance between tokens.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_relative_position: int = 64):
        super(RelativeMultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.max_relative_position = max_relative_position

        # Linear projections for Q, K, V.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Relative positional bias: shape (num_heads, 2 * max_relative_position - 1)
        self.relative_attention_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_relative_position - 1)
        )
        nn.init.xavier_uniform_(self.relative_attention_bias)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            key_padding_mask: Boolean tensor of shape (batch_size, seq_len) with True in padded positions.
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections and reshape: (batch_size, num_heads, seq_len, d_k)
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute standard dot product attention scores.
        # scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Compute relative position bias.
        # Create a matrix of relative positions (seq_len, seq_len)
        # Relative positions range from -(seq_len-1) to (seq_len-1)
        range_vec = torch.arange(seq_len, device=query.device)
        rel_pos = range_vec.unsqueeze(1) - range_vec.unsqueeze(0)  # (seq_len, seq_len)
        # Clip to the max_relative_position.
        rel_pos_clipped = torch.clamp(rel_pos, -self.max_relative_position + 1, self.max_relative_position - 1)
        # Shift values to be >= 0. New range: [0, 2*max_relative_position - 2]
        rel_pos_shifted = rel_pos_clipped + self.max_relative_position - 1  # (seq_len, seq_len)
        # Now, get the bias for each head: shape becomes (num_heads, seq_len, seq_len)
        relative_bias = self.relative_attention_bias[:, rel_pos_shifted]  # using advanced indexing

        # Add relative bias to the scores.
        scores = scores + relative_bias.unsqueeze(0)  # (batch_size, num_heads, seq_len, seq_len)

        # Apply key padding mask if provided.
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax over last dimension.
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Compute the attention output.
        # (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attn, V)
        # Concatenate heads: (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        return output


#########################################
# 3. Relative Transformer Encoder Layer
#########################################
class RelativeTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer that uses RelativeMultiheadAttention.
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 norm_first: bool = True):
        super(RelativeTransformerEncoderLayer, self).__init__()
        self.norm_first = norm_first
        self.self_attn = RelativeMultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.relu  # You can choose F.gelu if preferred

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = src
        if self.norm_first:
            src = self.norm1(src)
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = residual + self.dropout(src2)
        if not self.norm_first:
            src = self.norm1(src)

        residual = src
        if self.norm_first:
            src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout(src2)
        if not self.norm_first:
            src = self.norm2(src)
        return src


#########################################
# 4. Transformer Encoder Module with Relative Positional Encoding
#########################################
class TransformerCharEncoder(nn.Module):
    """
    Transformer-based character encoder with optional relative positional encoding.

    Args:
        input_size: Dimensionality of input features.
        embed_dim: Embedding dimension (must be divisible by num_heads).
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        projection_dim: Optional projection dimension for output (if None, returns embed_dim).
        max_len: Maximum sequence length for positional encodings.
        use_relative: If True, uses relative positional encoding in the self-attention mechanism.
        max_relative_position: Maximum relative distance to consider.
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
            use_relative: bool = True,
            max_relative_position: int = 64
    ):
        super(TransformerCharEncoder, self).__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.use_relative = use_relative

        # Project input features if necessary.
        if input_size != embed_dim:
            self.input_projection = nn.Linear(input_size, embed_dim)
        else:
            self.input_projection = nn.Identity()

        # Positional encoding (absolute); kept for compatibility.
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        self.input_dropout = nn.Dropout(dropout)

        # Build encoder layers.
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_relative:
                layer = RelativeTransformerEncoderLayer(
                    d_model=embed_dim,
                    num_heads=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=dropout,
                    norm_first=True
                )
            else:
                # Use PyTorch's built-in layer with pre-norm.
                layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True
                )
            self.layers.append(layer)
        self.norm = nn.LayerNorm(embed_dim)

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
        # Add absolute positional encodings.
        x = self.pos_encoder(x)
        x = self.input_dropout(x)

        # Create key padding mask: True for padded positions.
        device = inputs.device
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1)

        # Pass through the encoder layers.
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        # Optionally reduce dimension.
        x = self.reduce_dim(x)
        return x


#########################################
# 5. Sample Script to Test the Encoder
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
    use_relative = True
    max_relative_position = 64

    # Create a sample input tensor with random values.
    sample_input = torch.randn(batch_size, seq_len, input_size)
    # Define sequence lengths (e.g., first sequence has length 10, second sequence length 8).
    lengths = torch.tensor([10, 8])

    # Instantiate the TransformerCharEncoder with relative positional encoding.
    encoder = TransformerCharEncoder(
        input_size=input_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        projection_dim=projection_dim,
        max_len=max_len,
        use_relative=use_relative,
        max_relative_position=max_relative_position
    )

    # Pass the sample input through the encoder.
    output = encoder(sample_input, lengths)

    # Print output details.
    print("Output shape:", output.shape)
    print("Output tensor:", output)


if __name__ == '__main__':
    main()