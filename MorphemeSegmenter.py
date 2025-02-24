"""Does the morpheme segmentation as outlined in methodology"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Utilities import make_mask_2d, make_mask_3d
from torch.nn.functional import one_hot


class MorphemeSegmenter(nn.Module):
    neg_inf_val = -1e9

    def __init__(self, hidden_size: int, use_gumbel: bool = False, fixed_tau: float = None):
        """
        Args:
            hidden_size (int): Dimensionality of the encoder outputs.
            use_gumbel (bool): If True, use Gumbel-Softmax relaxation for differentiable sampling.
            fixed_tau (float, optional): If provided, use a fixed threshold instead of adaptive thresholding.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.use_gumbel = use_gumbel
        self.fixed_tau = fixed_tau

        # Linear layer to predict morpheme end scores (logits)
        self.morpheme_end_classifier = nn.Linear(self.hidden_size, 1)
        self.log_sigmoid = nn.LogSigmoid()
        self.cross_entropy = nn.CrossEntropyLoss()

        # Our additional component: an MLP to predict the adaptive threshold
        self.threshold_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()  # Outputs threshold in (0,1)
        )

    @staticmethod
    def get_best_paths(scores: Tensor, word_lengths: Tensor, num_morphemes: Tensor):
        # This function is the same as in Girrbach's implementation.
        # scores: shape [#words x #chars]
        num_words, num_chars = scores.shape

        # Compute Character -> Morpheme Mask
        max_num_morphemes = torch.max(num_morphemes).cpu().item()

        # Mask Separator Indices that Belong to Padding Chars
        index_mask = make_mask_2d(num_morphemes - 1)
        index_mask_padding = torch.ones(
            index_mask.shape[0], 1, dtype=torch.bool, device=index_mask.device
        )
        index_mask = torch.cat([index_mask, index_mask_padding], dim=1)
        index_mask = index_mask.to(scores.device)

        # Select Number of Separators (with the Highest Scores) according to Number of Morphemes
        # Because of Padding, Indices Start with 1 (We Remove 0 Later)
        best_separators = torch.topk(scores, dim=1, k=max_num_morphemes).indices
        best_separators = best_separators + 1
        best_separator_indices = torch.masked_fill(
            best_separators, mask=index_mask, value=0
        )

        # Convert Ordinal Indices to One-Hot Representations
        # e.g. [1, 4] -> [0, 1, 0, 0, 1, 0, 0] corresponds to 3 morphemes
        best_separators_one_hot = torch.zeros(
            num_words, num_chars + 1, dtype=torch.long, device=scores.device
        )
        best_separators_one_hot = torch.scatter(
            best_separators_one_hot, dim=1, index=best_separator_indices, value=1
        )
        # Remove Padding Indices
        best_separators_one_hot = best_separators_one_hot[:, 1:]
        # New Morpheme Starts at Next Character
        # -> Shift before cumsum
        best_separators_one_hot = torch.roll(best_separators_one_hot, shifts=1, dims=1)
        character_to_morpheme = best_separators_one_hot.cumsum(dim=1)

        # Mask Padding Characters (Avoid Appending to Last Morpheme)
        best_path_mask = make_mask_3d(word_lengths, num_morphemes)
        best_path_matrix = one_hot(character_to_morpheme, num_classes=max_num_morphemes)
        best_path_matrix = torch.masked_fill(
            best_path_matrix, mask=best_path_mask, value=0
        )
        best_path_matrix = best_path_matrix.bool()

        return best_path_matrix, best_separators_one_hot

    def get_marginals(self, scores: Tensor, word_lengths: Tensor, num_morphemes: Tensor):
        batch_size = scores.shape[0]
        max_num_chars = scores.shape[1]
        max_num_morphemes = torch.max(num_morphemes).cpu().item()

        log_sigmoid = self.log_sigmoid(scores)
        log_one_minus_sigmoid = self.log_sigmoid(-scores)

        beta_prior = torch.full(
            (batch_size, max_num_chars, max_num_morphemes),
            fill_value=self.neg_inf_val,
            device=scores.device,
        )
        beta_prior[torch.arange(batch_size), word_lengths - 1, num_morphemes - 1] = 0.0

        padding = torch.full((batch_size, 1), fill_value=self.neg_inf_val, device=scores.device)

        def pad_left(score_row: Tensor):
            return torch.cat([padding, score_row[:, :-1]], dim=1)

        def pad_right(score_row: Tensor):
            return torch.cat([score_row[:, 1:], padding], dim=1)

        alpha = [
            torch.full((batch_size, max_num_morphemes), fill_value=self.neg_inf_val, device=scores.device)
        ]
        alpha[0][:, 0] = 0.0

        for t in range(max_num_chars - 1):
            prev_alpha = alpha[-1]
            alpha_t_stay = prev_alpha + log_one_minus_sigmoid[:, t:t + 1]
            alpha_t_shift = pad_left(prev_alpha) + log_sigmoid[:, t:t + 1]
            alpha_t = torch.logaddexp(alpha_t_stay, alpha_t_shift)
            alpha.append(alpha_t)

        beta = [
            torch.full((batch_size, max_num_morphemes), fill_value=self.neg_inf_val, device=scores.device)
        ]
        for t in range(max_num_chars):
            t = max_num_chars - 1 - t
            next_beta = beta[0]
            beta_t_stay = next_beta + log_one_minus_sigmoid[:, t:t + 1]
            beta_t_shift = pad_right(next_beta) + log_sigmoid[:, t:t + 1]
            beta_t = torch.logaddexp(beta_t_stay, beta_t_shift)
            beta_t = torch.logaddexp(beta_t, beta_prior[:, t])
            beta.insert(0, beta_t)

        alpha = torch.stack(alpha, dim=1)
        beta = torch.stack(beta[:-1], dim=1)
        z = alpha[torch.arange(batch_size), word_lengths - 1, num_morphemes - 1]
        z = z.reshape(batch_size, 1, 1)

        marginal_mask = make_mask_3d(word_lengths, num_morphemes)
        marginals = (alpha + beta - z).exp()
        marginals = torch.masked_fill(marginals, mask=marginal_mask, value=0.0)

        return marginals

    def _select_relevant_morphemes(self, morpheme_encodings: Tensor, num_morphemes: Tensor) -> Tensor:
        """Select morpheme encodings that are not padding."""
        morpheme_encodings = morpheme_encodings.reshape(-1, self.hidden_size)
        morpheme_mask = make_mask_2d(num_morphemes).flatten()
        morpheme_mask = torch.logical_not(morpheme_mask)
        all_indices = torch.arange(morpheme_encodings.shape[0], device=morpheme_mask.device)
        selected_indices = torch.masked_select(all_indices, mask=morpheme_mask)
        morpheme_encodings = torch.index_select(morpheme_encodings, index=selected_indices, dim=0)
        return morpheme_encodings

    def forward(self,
                word_encodings: Tensor,
                word_lengths: Tensor,
                num_morphemes: Tensor = None,
                training: bool = False):
        """
        Args:
            word_encodings: Tensor of shape [#words x #chars x hidden_size].
            word_lengths: Tensor of shape [#words] with lengths.
            num_morphemes: Tensor specifying the number of morphemes per word (if available).
            training: Boolean flag to indicate training mode.
        Returns:
            For training: Soft morpheme representations computed via marginals.
            For inference: Hard morpheme representations via best path.
        """
        batch_size = word_encodings.shape[0]
        max_num_chars = word_encodings.shape[1]

        # Compute morpheme end scores (logits)
        score_mask = torch.ones(batch_size, max_num_chars, dtype=torch.bool)
        score_mask[:, :max_num_chars - 1] = make_mask_2d(word_lengths - 1)
        score_mask = score_mask.to(word_encodings.device)

        morpheme_end_scores = self.morpheme_end_classifier(word_encodings).squeeze(2)

        # Add Gaussian noise during training to encourage discreteness
        if training:
            morpheme_end_scores = morpheme_end_scores + torch.randn_like(morpheme_end_scores)
        morpheme_end_scores = torch.masked_fill(morpheme_end_scores, score_mask, value=self.neg_inf_val)

        # --- Our Adaptive Thresholding Improvement ---
        # Instead of using a fixed threshold, we compute an adaptive threshold per word.
        # Compute max-pooled representation over valid positions.
        mask = torch.arange(max_num_chars, device=word_encodings.device).unsqueeze(0) < word_lengths.unsqueeze(1)
        mask = mask.float().unsqueeze(-1)  # shape: (batch, seq_len, 1)
        pooled = (word_encodings * mask).max(dim=1)[0]  # (batch, hidden_size)
        adaptive_tau = self.threshold_predictor(pooled).squeeze(-1)  # (batch,)
        # If fixed_tau is provided, override adaptive threshold.
        if self.fixed_tau is not None:
            tau = torch.full((batch_size,), self.fixed_tau, device=word_encodings.device)
        else:
            tau = adaptive_tau

        # --- End Adaptive Thresholding ---

        # Depending on the training flag and use_gumbel setting,
        # choose a method to compute the segmentation decisions.
        if training and self.use_gumbel:
            # Use Gumbel-Softmax for a differentiable hard decision.
            logits = torch.stack([morpheme_end_scores, -morpheme_end_scores], dim=-1)
            seg_sample = F.gumbel_softmax(logits, tau=0.5, hard=True)
            seg_decisions = seg_sample[:, :, 0]  # shape: (batch, seq_len)
        else:
            # Use standard thresholding: a boundary is marked if seg_prob > tau.
            seg_probs = torch.sigmoid(morpheme_end_scores)
            seg_decisions = (seg_probs > tau.unsqueeze(1)).float()

        # Get best path matrix from segmentation scores.
        best_path_matrix, _ = self.get_best_paths(morpheme_end_scores, word_lengths, num_morphemes)
        best_path_matrix = best_path_matrix.to(morpheme_end_scores.device)

        if not training:
            # In inference mode, use hard attention directly.
            word_encodings_t = word_encodings.transpose(1, 2)
            morpheme_encodings = torch.bmm(word_encodings_t, best_path_matrix.float())
            morpheme_encodings = morpheme_encodings.transpose(1, 2)
            morpheme_encodings = self._select_relevant_morphemes(morpheme_encodings, num_morphemes)
            return morpheme_encodings, best_path_matrix

        # Otherwise, compute marginals for soft segmentation
        marginals = self.get_marginals(morpheme_end_scores, word_lengths, num_morphemes)
        # Compute soft morpheme representations.
        word_encodings_t = word_encodings.transpose(1, 2)
        morpheme_encodings = torch.bmm(word_encodings_t, marginals)
        morpheme_encodings = morpheme_encodings.transpose(1, 2)
        # Compute residuals (for the straight-through estimator)
        residual_scores = torch.where(best_path_matrix, marginals - 1.0, marginals)
        residuals = torch.bmm(word_encodings_t, residual_scores)
        residuals = residuals.transpose(1, 2).detach()
        morpheme_encodings = morpheme_encodings - residuals
        # Select morphemes that are not padding.
        morpheme_encodings = self._select_relevant_morphemes(morpheme_encodings, num_morphemes)
        return morpheme_encodings, best_path_matrix

# Testing the module
if __name__ == "__main__":
    # Set parameters for the test
    batch_size = 4  # Number of words in the batch
    seq_len = 10  # Padded sequence length (number of characters per word)
    hidden_size = 256  # Dimensionality of the encoder outputs

    # Simulate encoder outputs: these mimic the output from our Transformer encoder
    dummy_encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)

    # Simulate word lengths (for example, each word may have fewer than 10 valid characters)
    word_lengths = torch.tensor([10, 8, 9, 7])

    # Simulate the expected number of morphemes per word (this might come from gold data or estimates)
    num_morphemes = torch.tensor([3, 2, 2, 2])

    # Instantiate the improved morpheme segmenter
    # Here, use_gumbel is set to False to use standard adaptive thresholding.
    segmenter = MorphemeSegmenter(hidden_size, use_gumbel=False, fixed_tau=None)

    # ----- Run in Training Mode (Soft Segmentation) -----
    segmenter.train()  # Set module to training mode
    soft_outputs = segmenter(dummy_encoder_outputs, word_lengths, num_morphemes, training=True)
    morpheme_encodings_soft, best_path_matrix_soft = soft_outputs

    print("==== Training Mode (Soft Segmentation) ====")
    print("Morpheme Encodings (soft):")
    print(morpheme_encodings_soft)
    print("\nBest Path Matrix (soft):")
    print(best_path_matrix_soft)

    # ----- Run in Inference Mode (Hard Segmentation) -----
    segmenter.eval()  # Set module to evaluation mode
    with torch.no_grad():
        hard_outputs = segmenter(dummy_encoder_outputs, word_lengths, num_morphemes, training=False)
    morpheme_encodings_hard, best_path_matrix_hard = hard_outputs

    print("\n==== Inference Mode (Hard Segmentation) ====")
    print("Morpheme Encodings (hard):")
    print(morpheme_encodings_hard)
    print("\nBest Path Matrix (hard):")
    print(best_path_matrix_hard)
