"""Extra functions for utility and convenience to be incorporated into our pipeline"""

import torch

def make_mask(seq_len, lengths):
    """
    Creates a boolean mask for a batch of sequences.
    Args:
        seq_len (int): Maximum sequence length.
        lengths (Tensor): Tensor of shape (batch,) with valid lengths.
    Returns:
        mask (Tensor): Boolean mask of shape (batch, seq_len) where True indicates a valid position.
    """
    batch_size = lengths.size(0)
    mask = torch.arange(seq_len, device=lengths.device).unsqueeze(0).expand(batch_size, seq_len)
    return mask < lengths.unsqueeze(1)

def make_mask_2d(lengths):
    """
    Given a tensor of lengths, returns a 2D mask (batch x max_length).
    """
    max_len = lengths.max().item()
    return make_mask(max_len, lengths)

def make_mask_3d(word_lengths, num_morphemes):
    """
    (Optional) Returns a 3D mask for structured predictions.
    For now, this is a placeholder.
    """
    batch_size = word_lengths.size(0)
    max_len = word_lengths.max().item()
    max_morphemes = num_morphemes.max().item()
    return torch.zeros(batch_size, max_len, max_morphemes, dtype=torch.bool, device=word_lengths.device)

def max_pool_2d(x: torch.Tensor, lengths: torch.Tensor):
    # x: shape [batch x timesteps x features]
    mask = make_mask_2d(lengths).to(x.device).unsqueeze(-1)
    x = torch.masked_fill(x, mask=mask, value=-1e9)
    x = torch.max(x, dim=1).values
    return x