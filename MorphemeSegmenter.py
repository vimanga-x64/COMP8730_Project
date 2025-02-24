"""Does the morpheme segmentation as outlined in methodology"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utilities import make_mask, make_mask_2d, make_mask_3d


class MorphemeSegmenter(nn.Module):
    def __init__(self, hidden_size, use_gumbel=False, fixed_tau=None):
        """
        Args:
            hidden_size (int): Dimension of the encoder's hidden states.
            use_gumbel (bool): If True, use Gumbel-Softmax for differentiable sampling.
            fixed_tau (float or None): If provided, a fixed threshold is used instead of adaptive thresholding.
        """
        super(MorphemeSegmenter, self).__init__()
        self.hidden_size = hidden_size
        self.use_gumbel = use_gumbel
        self.fixed_tau = fixed_tau

        # Linear layer to compute segmentation logits for each character
        self.segmentation_linear = nn.Linear(hidden_size, 1)

        # MLP to predict an adaptive threshold from the max-pooled encoder outputs
        self.threshold_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, encoder_outputs, lengths):
        batch_size, seq_len, hidden_size = encoder_outputs.size()

        # Compute segmentation logits and probabilities.
        seg_logits = self.segmentation_linear(encoder_outputs).squeeze(-1)  # (batch, seq_len)
        seg_probs = torch.sigmoid(seg_logits)  # (batch, seq_len)

        # Always compute the adaptive threshold tau.
        if self.fixed_tau is not None:
            tau = torch.full((batch_size,), self.fixed_tau, device=encoder_outputs.device)
        else:
            mask = make_mask(seq_len, lengths).float()  # (batch, seq_len)
            # Multiply by mask (or use a masked max operation) to get a pooled representation.
            masked_enc_outputs = encoder_outputs * mask.unsqueeze(-1)
            pooled, _ = masked_enc_outputs.max(dim=1)  # (batch, hidden_size)
            tau = self.threshold_predictor(pooled).squeeze(-1)  # (batch,)

        # Decide on segmentation using either Gumbel-Softmax or thresholding.
        if self.use_gumbel and self.training:
            # For binary decision, stack logits for two classes: boundary and no-boundary.
            logits = torch.stack([seg_logits, -seg_logits], dim=-1)  # (batch, seq_len, 2)
            seg_sample = F.gumbel_softmax(logits, tau=0.5, hard=True)
            seg_decisions = seg_sample[:, :, 0]  # Use the first column as decision.
        else:
            seg_decisions = (seg_probs > tau.unsqueeze(1)).float()

        refined_segmentation = seg_decisions  # Placeholder for further structured refinement.

        return refined_segmentation, seg_probs, tau


# Testing the module
if __name__ == "__main__":
    # Create dummy encoder outputs: (batch_size, seq_len, hidden_size)
    batch_size = 4
    seq_len = 10
    hidden_size = 256
    dummy_encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
    # Dummy lengths for each sequence (assume varying sequence lengths)
    lengths = torch.tensor([10, 8, 9, 7])

    segmenter = MorphemeSegmenter(hidden_size, use_gumbel=True, fixed_tau=None)
    refined_segmentation, seg_probs, tau = segmenter(dummy_encoder_outputs, lengths)

    print("Refined segmentation decisions:\n", refined_segmentation)
    print("Segmentation probabilities:\n", seg_probs)
    print("Adaptive thresholds:\n", tau)