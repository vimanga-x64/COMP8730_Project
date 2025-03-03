"""Improved Unsupervised Morpheme Segmenter Module with Full Forward-Backward and Utility Masks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.functional import one_hot
from Utilities import make_mask_2d, make_mask_3d


class MorphemeSegmenter(nn.Module):
    """
    Computes segmentation probabilities for each character and makes hard decisions.
    Improvements include:
      1. Structured marginalization via a full forward-backward algorithm.
      2. Multi-task outputs (segmentation mask and morpheme count) for joint training.
      3. Self-attention to enhance contextual segmentation decisions.
      4. Enhanced adaptive thresholding using max, mean, and variance of encoder outputs.
      5. Incorporation of utility masking functions (make_mask_2d and make_mask_3d)
         to handle variable-length sequences.
    """
    neg_inf_val = -1e9

    def __init__(self, embed_dim: int, use_gumbel: bool = False, temperature: float = 1.0, use_attention: bool = True,
                 fixed_K: int = 5):
        super(MorphemeSegmenter, self).__init__()
        self.hidden_size = embed_dim
        self.seg_classifier = nn.Linear(embed_dim, 1)
        # Enhanced adaptive thresholding: input richer statistics (max, mean, var).
        self.threshold_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim // 2),
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

        # Attention-based segmentation: refine encoder outputs with self-attention.
        self.use_attention = use_attention
        if self.use_attention:
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # Fixed number of segments (K) to use in the forward-backward algorithm.
        self.fixed_K = fixed_K

    def _forward_backward(self, seg_probs: torch.Tensor, L: int, K: int) -> (torch.Tensor, torch.Tensor):
        """
        Performs forward-backward algorithm in log-space for one instance with valid length L.
        Args:
            seg_probs: Tensor of shape (seq_len,) with segmentation probabilities.
            L: Integer, valid length (number of characters) in the instance.
            K: Fixed number of segments.
        Returns:
            alpha: Tensor of shape (L+1, K+1) with forward log probabilities.
            beta: Tensor of shape (L+1, K+1) with backward log probabilities.
        """
        valid_probs = seg_probs[:L]  # consider only valid positions
        eps = 1e-10
        log_p = torch.log(valid_probs + eps)
        log_not_p = torch.log(1 - valid_probs + eps)

        # Initialize alpha: shape (L+1, K+1)
        alpha = torch.full((L + 1, K + 1), -float('inf'), device=seg_probs.device)
        alpha[0, 0] = 0.0
        neg_inf_tensor = torch.tensor(-float('inf'), device=seg_probs.device)
        for i in range(1, L + 1):
            for j in range(0, min(i, K + 1)):
                opt1 = alpha[i - 1, j] + log_not_p[i - 1]
                opt2 = alpha[i - 1, j - 1] + log_p[i - 1] if j > 0 else neg_inf_tensor
                alpha[i, j] = torch.logsumexp(torch.stack([opt1, opt2]), dim=0)

        # Initialize beta: shape (L+1, K+1)
        beta = torch.full((L + 1, K + 1), -float('inf'), device=seg_probs.device)
        beta[L, K] = 0.0
        for i in reversed(range(L)):
            for j in range(0, min(i + 1, K + 1)):
                opt1 = beta[i + 1, j] + log_not_p[i]
                opt2 = beta[i + 1, j + 1] + log_p[i] if (j + 1) <= K else neg_inf_tensor
                beta[i, j] = torch.logsumexp(torch.stack([opt1, opt2]), dim=0)
        return alpha, beta

    def get_marginals(self, seg_probs: torch.Tensor, L: int, K: int) -> torch.Tensor:
        """
        Computes marginal probability for each segmentation boundary for a single instance.
        Args:
            seg_probs: Tensor of shape (seq_len,) with raw segmentation probabilities.
            L: Integer, valid length of the sequence.
            K: Fixed number of segments.
        Returns:
            marginals_full: Tensor of shape (seq_len,) with marginal probabilities.
        """
        alpha, beta = self._forward_backward(seg_probs, L, K)
        Z = alpha[L, K]  # total log probability (scalar)
        marginals_valid = torch.full((L,), 0.0, device=seg_probs.device)
        for i in range(L):
            log_sum = -float('inf')
            for j in range(1, K + 1):
                log_sum = torch.logsumexp(torch.tensor([log_sum,
                                                        alpha[i, j - 1] + torch.log(seg_probs[i] + 1e-10) + beta[
                                                            i + 1, j]],
                                                       device=seg_probs.device), dim=0)
            marginals_valid[i] = torch.exp(log_sum - Z)

        # Pad marginals to the full sequence length.
        seq_len = seg_probs.size(0)
        if L < seq_len:
            padded = torch.zeros(seq_len - L, device=seg_probs.device)
            marginals_full = torch.cat([marginals_valid, padded], dim=0)
        else:
            marginals_full = marginals_valid

        # Use make_mask_2d to create a valid mask for the full sequence.
        valid_mask = make_mask_2d(torch.tensor([L], device=seg_probs.device))[0]  # shape (L,)
        # Now pad the valid_mask to length seq_len.
        if L < seq_len:
            pad_mask = torch.zeros(seq_len - L, dtype=torch.bool, device=seg_probs.device)
            valid_mask = torch.cat([valid_mask, pad_mask], dim=0)
        marginals_full = marginals_full * valid_mask.float()
        return marginals_full

    def forward(self, encoder_outputs: torch.Tensor, word_lengths: torch.Tensor, num_morphemes: torch.Tensor = None,
                training: bool = False):
        """
        Args:
            encoder_outputs: Tensor of shape (batch_size, seq_len, embed_dim)
            word_lengths: Tensor of shape (batch_size,) indicating the valid length for each word.
            num_morphemes: Tensor of shape (batch_size,) with the target number of morphemes per word (if available).
            training: Boolean flag; if True, use hard segmentation (with straight-through estimation).
        Returns:
            segmentation_mask: Binary segmentation decisions (batch_size x seq_len)
            morpheme_count: Predicted morpheme count (batch_size x 1)
            tau: Adaptive threshold values (batch_size x 1)
            seg_probs: Raw segmentation probabilities (batch_size x seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.size()

        # Optionally refine encoder outputs with self-attention.
        if self.use_attention:
            attn_output, _ = self.self_attn(encoder_outputs, encoder_outputs, encoder_outputs)
            encoder_outputs = encoder_outputs + attn_output  # Residual connection.

        # Compute segmentation logits and raw probabilities.
        seg_logits = self.seg_classifier(encoder_outputs).squeeze(-1)  # (batch_size, seq_len)
        seg_probs = torch.sigmoid(seg_logits)

        if self.use_gumbel and training:
            noise = -torch.log(-torch.log(torch.rand_like(seg_logits) + 1e-10) + 1e-10)
            seg_logits = (seg_logits + noise) / self.temperature
            seg_probs = torch.sigmoid(seg_logits)

        # Enhanced adaptive thresholding: combine max, mean, and variance.
        z_max, _ = torch.max(encoder_outputs, dim=1)  # (batch_size, embed_dim)
        z_mean = torch.mean(encoder_outputs, dim=1)  # (batch_size, embed_dim)
        z_var = torch.var(encoder_outputs, dim=1)  # (batch_size, embed_dim)
        z_cat = torch.cat([z_max, z_mean, z_var], dim=-1)  # (batch_size, 3*embed_dim)
        tau = torch.sigmoid(self.threshold_mlp(z_cat))  # (batch_size, 1)

        # Predict morpheme count (auxiliary output) using mean representation.
        morpheme_count = F.softplus(self.count_mlp(z_mean))  # (batch_size, 1)

        # Compute marginals for each instance, taking into account variable lengths.
        marginals_list = []
        for b in range(batch_size):
            L = word_lengths[b].item()  # valid length for instance b
            seg_probs_b = seg_probs[b, :]  # (seq_len,)
            marginals_b = self.get_marginals(seg_probs_b, L, self.fixed_K)
            marginals_list.append(marginals_b)
        marginals = torch.stack(marginals_list, dim=0)  # (batch_size, seq_len)

        # Optionally create a 3D mask using make_mask_3d if available.
        if num_morphemes is not None:
            mask_3d = make_mask_3d(word_lengths, num_morphemes).to(encoder_outputs.device)
            marginals = torch.masked_fill(marginals.unsqueeze(-1), mask=mask_3d, value=0.0).squeeze(-1)

        # Use hard thresholding with straight-through estimation if training.
        if training:
            hard_mask = (marginals > tau).float()
            segmentation_mask = hard_mask.detach() - marginals.detach() + marginals
        else:
            segmentation_mask = marginals

        return segmentation_mask, morpheme_count, tau, seg_probs


if __name__=="__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Sample parameters
    batch_size = 4    # Number of words in the batch
    seq_len = 10      # Padded sequence length (number of characters per word)
    embed_dim = 256   # Dimensionality of the encoder outputs

    # Create dummy encoder outputs (simulate character-level encoder output)
    dummy_encoder_outputs = torch.randn(batch_size, seq_len, embed_dim)

    # Simulate word lengths (for example, each word may have fewer than 10 valid characters)
    word_lengths = torch.tensor([10, 8, 9, 7])

    # Simulate the expected number of morphemes per word (this might come from gold data or estimates)
    num_morphemes = torch.tensor([3, 2, 2, 2])

    # Instantiate the morpheme segmenter (Gumbel noise enabled)
    segmenter = MorphemeSegmenter(embed_dim, use_gumbel=True, temperature=1.0, use_attention=True, fixed_K=5)

    print("=== Running in Training Mode (learn_segmentation=True) ===")
    segmenter.train()  # Set module to training mode
    seg_mask_train, morpheme_count_train, tau_train, seg_probs_train = segmenter(dummy_encoder_outputs, word_lengths, num_morphemes, training=True)
    print("Segmentation Mask (Training Mode):")
    print(seg_mask_train)
    print("Predicted Morpheme Count:")
    print(morpheme_count_train)
    print("Adaptive Threshold (tau):")
    print(tau_train)
    print("Raw Segmentation Probabilities:")
    print(seg_probs_train)

    print("\n=== Running in Inference Mode (learn_segmentation=False) ===")
    segmenter.eval()  # Set module to evaluation mode
    with torch.no_grad():
        seg_mask_infer, morpheme_count_infer, tau_infer, seg_probs_infer = segmenter(dummy_encoder_outputs, word_lengths, num_morphemes, training=False)
    print("Segmentation Mask (Inference Mode):")
    print(seg_mask_infer)
    print("Predicted Morpheme Count:")
    print(morpheme_count_infer)
    print("Adaptive Threshold (tau):")
    print(tau_infer)
    print("Raw Segmentation Probabilities:")
    print(seg_probs_infer)