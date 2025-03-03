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
    Configured to use batch_first=True.
    """

    def __init__(self, gloss_vocab_size: int, embed_dim: int, num_heads: int,
                 ff_dim: int, num_layers: int, dropout: float = 0.1, max_len: int = 5000, tie_weights: bool = False):
        super(GlossingDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.tie_weights = tie_weights
        self.embedding = nn.Embedding(gloss_vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len)
        # Create a Transformer decoder layer with batch_first=True.
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, gloss_vocab_size)

        # Optionally tie the weights.
        if self.tie_weights:
            self.fc_out.weight = self.embedding.weight

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate a square mask for the target sequence. Masked positions are filled with -inf,
        unmasked positions with 0.0.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                return_attn: bool = False) -> torch.Tensor:
        """
        Args:
            tgt: Target gloss token indices (batch_size, tgt_seq_len)
            memory: Aggregated segment representations (batch_size, mem_len, embed_dim)
            tgt_mask: Optional mask for target sequence.
            memory_key_padding_mask: Optional mask for memory.
            return_attn: If True, returns attention weights (requires custom modifications).
        Returns:
            logits: Tensor of shape (batch_size, tgt_seq_len, gloss_vocab_size)
        """
        # Scale embeddings.
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.embed_dim)
        tgt_embedded = self.pos_encoding(tgt_embedded)

        # If no target mask is provided, generate a causal mask.
        if tgt_mask is None:
            tgt_seq_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, tgt.device)

        # Forward pass through the decoder.
        # Note: Standard TransformerDecoder layers do not return attention weights.
        decoded = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask,
                               memory_key_padding_mask=memory_key_padding_mask)

        logits = self.fc_out(decoded)
        """
        to be experimented with: 
        
        # If attention weights are needed, further modifications to the decoder layers are required.
        if return_attn:
            # Placeholder: Returning logits only.
            # To return attention weights, you would need to modify the decoder layer to capture and output them.
            return logits, None
            
        """
        return logits


def main():
    # Dummy parameters.
    batch_size = 2
    tgt_seq_len = 5  # Length of target sequence (e.g., gloss tokens)
    mem_len = 7  # Length of memory (aggregated segment representations)
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_layers = 2
    dropout = 0.1
    gloss_vocab_size = 10  # Dummy vocabulary size
    max_len = 50  # Maximum length for positional encoding
    tie_weights = True

    # Instantiate the GlossingDecoder.
    decoder = GlossingDecoder(
        gloss_vocab_size=gloss_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_len=max_len,
        tie_weights=tie_weights
    )

    # Create dummy target token indices.
    tgt = torch.randint(0, gloss_vocab_size, (batch_size, tgt_seq_len))
    # Create dummy memory tensor (e.g., aggregated segment representations).
    memory = torch.randn(batch_size, mem_len, embed_dim)

    # Forward pass through the decoder.
    logits = decoder(tgt, memory)

    # Log output shapes.
    print("Dummy target indices:")
    print(tgt)
    print("\nOutput logits shape:", logits.shape)
    print("\nOutput logits:")
    print(logits)


if __name__ == "__main__":
    main()