"""Does the morpheme segmentation as outlined in methodology"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Utilities import make_mask_2d, make_mask_3d
from torch.nn.functional import one_hot

class MorphemeSegmenter(nn.Module):
    """
    Computes segmentation probabilities for each character and makes hard decisions.
    Optionally uses Gumbel-Softmax relaxation for smoother gradients.
    """

    def __init__(self, embed_dim: int, use_gumbel: bool = False, temperature: float = 1.0):
        super(MorphemeSegmenter, self).__init__()
        self.W_seg = nn.Linear(embed_dim, 1)
        self.threshold_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.count_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.use_gumbel = use_gumbel
        self.temperature = temperature

    def forward(self, encoder_outputs: torch.Tensor, learn_segmentation: bool = True):
        """
        Args:
            encoder_outputs: Tensor of shape (batch_size, seq_len, embed_dim)
            learn_segmentation: If False, return soft probabilities.
        Returns:
            segmentation_mask: Binary segmentation decisions (batch_size, seq_len)
            morpheme_count: Predicted morpheme count (batch_size, 1)
            tau: Adaptive threshold (batch_size, 1)
            seg_probs: Raw segmentation probabilities (batch_size, seq_len)
        """
        seg_logits = self.W_seg(encoder_outputs).squeeze(-1)  # (batch_size, seq_len)
        seg_probs = torch.sigmoid(seg_logits)

        if self.use_gumbel and learn_segmentation:
            noise = -torch.log(-torch.log(torch.rand_like(seg_logits) + 1e-10) + 1e-10)
            seg_logits = (seg_logits + noise) / self.temperature
            seg_probs = torch.sigmoid(seg_logits)

        # Max-pool encoder outputs to get word-level representation.
        z, _ = torch.max(encoder_outputs, dim=1)  # (batch_size, embed_dim)
        tau = torch.sigmoid(self.threshold_mlp(z))  # (batch_size, 1)
        morpheme_count = F.softplus(self.count_mlp(z))  # (batch_size, 1)

        if learn_segmentation:
            hard_mask = (seg_probs > tau).float()
            segmentation_mask = hard_mask.detach() - seg_probs.detach() + seg_probs
        else:
            segmentation_mask = seg_probs

        return segmentation_mask, morpheme_count, tau, seg_probs