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
    batch_size = word_lengths.size(0)
    max_len = word_lengths.max().item()
    # Use 1 instead of max_morphemes so that the mask shape is (batch_size, max_len, 1)
    return torch.zeros(batch_size, max_len, 1, dtype=torch.bool, device=word_lengths.device)

def max_pool_2d(x: torch.Tensor, lengths: torch.Tensor):
    # x: shape [batch x timesteps x features]
    mask = make_mask_2d(lengths).to(x.device).unsqueeze(-1)
    x = torch.masked_fill(x, mask=mask, value=-1e9)
    x = torch.max(x, dim=1).values
    return x


def aggregate_segments(encoder_outputs: torch.Tensor, segmentation_mask: torch.Tensor) -> torch.Tensor:
    """
    Aggregates encoder outputs into morpheme-level representations using segmentation boundaries.

    Args:
        encoder_outputs: Tensor of shape (batch_size, seq_len, embed_dim) containing encoder outputs.
        segmentation_mask: Tensor of shape (batch_size, seq_len) with binary values (1 indicates a boundary).

    Returns:
        seg_tensor: Tensor of shape (batch_size, max_segments, embed_dim) containing averaged morpheme representations.
    """
    batch_size, seq_len, embed_dim = encoder_outputs.size()
    segments = []  # List to store segments for each word in the batch
    num_segments_list = []

    for b in range(batch_size):
        word_enc = encoder_outputs[b]  # (seq_len, embed_dim)
        seg_mask = segmentation_mask[b]  # (seq_len,)
        seg_reps = []
        start = 0
        for i in range(seq_len):
            # Check if current character is a boundary
            if seg_mask[i] >= 0.5:
                # Aggregate characters from start to i (inclusive)
                if i >= start:  # Ensure non-empty segment
                    seg_rep = word_enc[start:i + 1].mean(dim=0)
                    seg_reps.append(seg_rep)
                start = i + 1
        # If any characters remain after the last boundary, aggregate them
        if start < seq_len:
            seg_rep = word_enc[start:seq_len].mean(dim=0)
            seg_reps.append(seg_rep)
        # If no boundaries were detected, fall back to a single segment (average of entire word)
        if len(seg_reps) == 0:
            seg_reps.append(word_enc.mean(dim=0))
        seg_reps = torch.stack(seg_reps, dim=0)  # (num_segments, embed_dim)
        segments.append(seg_reps)
        num_segments_list.append(seg_reps.size(0))

    # Pad all segment tensors to the maximum number of segments in the batch.
    max_segments = max(num_segments_list)
    seg_tensor = torch.zeros(batch_size, max_segments, embed_dim, device=encoder_outputs.device)
    for b in range(batch_size):
        segs = segments[b]
        seg_tensor[b, :segs.size(0), :] = segs
    return seg_tensor