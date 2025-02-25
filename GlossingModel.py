import torch
import torch.nn as nn
from Encoder import TransformerCharEncoder, PositionalEncoding
from MorphemeSegmenter import MorphemeSegmenter
from GlossingDecoder import GlossingDecoder
from Utilities import aggregate_segments


#########################################
# 6. Overall Glossing Pipeline Model
#########################################
class GlossingPipeline(nn.Module):
    """
    Full pipeline that integrates:
      - Transformer-based character encoder,
      - Improved morpheme segmentation with adaptive thresholding,
      - A translation encoder to incorporate translation information,
      - Glossing decoder with cross-attention over memory augmented with translation.

    The segmentation learning can be toggled (learn_segmentation) for track 1 data.
    """

    def __init__(self, char_vocab_size: int, gloss_vocab_size: int, trans_vocab_size: int,
                 embed_dim: int = 256, num_heads: int = 8, ff_dim: int = 512,
                 num_layers: int = 6, dropout: float = 0.1, use_gumbel: bool = False):
        super(GlossingPipeline, self).__init__()
        # The encoder expects input features of dimension char_vocab_size.
        # Here we assume one-hot representations, so input_size == char_vocab_size.
        self.encoder = TransformerCharEncoder(
            input_size=char_vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            projection_dim=None
        )
        self.segmentation = MorphemeSegmenter(embed_dim, use_gumbel=use_gumbel)
        self.decoder = GlossingDecoder(
            gloss_vocab_size=gloss_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        # Translation encoder: simple embedding + mean pooling.
        self.translation_encoder = nn.Embedding(trans_vocab_size, embed_dim)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, tgt: torch.Tensor, trans: torch.Tensor,
                learn_segmentation: bool = True):
        """
        Args:
            src: Source word as sequence of character features (batch_size, src_seq_len, char_vocab_size)
            src_lengths: Tensor (batch_size,) with valid lengths.
            tgt: Target gloss token indices (batch_size, tgt_seq_len)
            trans: Translation token indices (batch_size, trans_seq_len)
            learn_segmentation: Whether to learn segmentation (True for track 1 data).
        Returns:
            logits: Tensor (batch_size, tgt_seq_len, gloss_vocab_size)
            morpheme_count: Predicted morpheme counts (batch_size, 1)
            tau: Adaptive threshold (batch_size, 1)
            seg_probs: Raw segmentation probabilities (batch_size, src_seq_len)
        """
        # Encode the source characters.
        encoder_outputs = self.encoder(src, src_lengths)  # (batch_size, src_seq_len, embed_dim)
        # Compute segmentation.
        segmentation_mask, morpheme_count, tau, seg_probs = self.segmentation(encoder_outputs, learn_segmentation)
        # Aggregate encoder outputs into segments.
        seg_tensor = aggregate_segments(encoder_outputs, segmentation_mask)  # (batch_size, num_segments, embed_dim)
        # Encode the translation and get a representation (mean pooling).
        trans_embedded = self.translation_encoder(trans)  # (batch_size, trans_seq_len, embed_dim)
        trans_repr = trans_embedded.mean(dim=1, keepdim=True)  # (batch_size, 1, embed_dim)
        # Prepend translation representation to the segment memory.
        memory = torch.cat([trans_repr, seg_tensor], dim=1)  # (batch_size, num_segments+1, embed_dim)
        # Decode gloss tokens using cross-attention over the memory.
        logits = self.decoder(tgt, memory)
        return logits, morpheme_count, tau, seg_probs

