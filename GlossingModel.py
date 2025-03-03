import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import TransformerCharEncoder
from MorphemeSegmenter import MorphemeSegmenter
from GlossingDecoder import GlossingDecoder
from Utilities import aggregate_segments  # Import aggregate_segments from Utilities
import pytorch_lightning as pl


#########################################
# 6. Integrated Glossing Pipeline as a LightningModule
#########################################
class GlossingPipeline(pl.LightningModule):
    """
    An integrated glossing pipeline that combines:
      - A Transformer-based character encoder,
      - An improved morpheme segmentation module with adaptive thresholding,
      - A translation encoder,
      - A glossing decoder with cross-attention over aggregated segment representations.

    This module is a PyTorch LightningModule so that training, validation,
    and optimizer configuration are integrated.
    """

    def __init__(self, char_vocab_size: int, gloss_vocab_size: int, trans_vocab_size: int,
                 embed_dim: int = 256, num_heads: int = 8, ff_dim: int = 512,
                 num_layers: int = 6, dropout: float = 0.1, use_gumbel: bool = False,
                 learning_rate: float = 0.001, gloss_pad_idx: int = None):
        super(GlossingPipeline, self).__init__()
        self.save_hyperparameters(ignore=["gloss_pad_idx"])
        # Build model components.
        self.encoder = TransformerCharEncoder(
            input_size=char_vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            projection_dim=None
        )
        # For Track 1 (unsupervised segmentation), gold segmentation is not available so pass None.
        self.segmentation = MorphemeSegmenter(embed_dim, use_gumbel=use_gumbel)
        self.decoder = GlossingDecoder(
            gloss_vocab_size=gloss_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.translation_encoder = nn.Embedding(trans_vocab_size, embed_dim)
        # Loss function.
        self.criterion = nn.CrossEntropyLoss(ignore_index=gloss_pad_idx)
        self.learning_rate = learning_rate

    def forward(self, src_features, src_lengths, tgt, trans, learn_segmentation: bool = True, num_morphemes=None):
        """
        Forward pass through the glossing pipeline.
        Args:
            src_features: Source character features (batch_size, src_seq_len, char_vocab_size) as one-hot.
            src_lengths: Valid lengths of source sequences (batch_size,).
            tgt: Target gloss token indices (batch_size, tgt_seq_len).
            trans: Translation token indices (batch_size, trans_seq_len).
            learn_segmentation: Whether to learn segmentation (True for Track 1 data).
            num_morphemes: If available (Track 2), the target number of morphemes per word;
                           set to None for unsupervised segmentation (Track 1).
        Returns:
            logits, morpheme_count, tau, seg_probs.
        """
        # Encode source characters.
        encoder_outputs = self.encoder(src_features, src_lengths)

        # Compute segmentation.
        # For Track 1 (unsupervised), num_morphemes is None.
        segmentation_mask, morpheme_count, tau, seg_probs = self.segmentation(
            encoder_outputs, src_lengths, num_morphemes, training=learn_segmentation
        )

        # Aggregate encoder outputs into morpheme representations.
        seg_tensor = aggregate_segments(encoder_outputs, segmentation_mask)

        # Encode translation and get representation.
        trans_embedded = self.translation_encoder(trans)  # (batch_size, trans_seq_len, embed_dim)
        trans_repr = trans_embedded.mean(dim=1, keepdim=True)  # (batch_size, 1, embed_dim)

        # Prepend translation representation to the segment memory.
        memory = torch.cat([trans_repr, seg_tensor], dim=1)

        # Decode gloss tokens using cross-attention.
        logits = self.decoder(tgt, memory)
        return logits, morpheme_count, tau, seg_probs

    def training_step(self, batch, batch_idx):
        src_batch, src_len_batch, tgt_batch, trans_batch = batch
        # Convert source indices into one-hot vectors.
        src_features = F.one_hot(src_batch, num_classes=self.encoder.input_size).float()
        # For Track 1, no gold segmentation is available, so pass None for num_morphemes.
        logits, morpheme_count, tau, seg_probs = self(src_features, src_len_batch, tgt_batch, trans_batch,
                                                      learn_segmentation=True, num_morphemes=None)
        # logits: (batch_size, tgt_seq_len, gloss_vocab_size)
        batch_size, tgt_seq_len, gloss_vocab_size = logits.size()
        logits = logits.view(-1, gloss_vocab_size)
        tgt_flat = tgt_batch.view(-1)
        loss = self.criterion(logits, tgt_flat)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)