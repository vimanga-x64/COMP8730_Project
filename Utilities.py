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


#########################################
# 4. Aggregation Function
#########################################
def aggregate_segments(encoder_outputs: torch.Tensor, segmentation_mask: torch.Tensor):
    """
    Aggregates encoder outputs into segment representations using segmentation boundaries.

    Args:
        encoder_outputs: Tensor (batch_size, seq_len, embed_dim)
        segmentation_mask: Tensor (batch_size, seq_len) with 1 indicating a boundary.
    Returns:
        seg_tensor: Tensor (batch_size, max_segments, embed_dim) with averaged segment representations.
    """
    batch_size, seq_len, embed_dim = encoder_outputs.size()
    segments = []
    for b in range(batch_size):
        outputs = encoder_outputs[b]  # (seq_len, embed_dim)
        mask = segmentation_mask[b]  # (seq_len)
        seg_reps = []
        start = 0
        for i in range(seq_len):
            if mask[i] >= 0.5:
                if i - start + 1 > 0:
                    seg_rep = outputs[start:i + 1].mean(dim=0)
                    seg_reps.append(seg_rep)
                start = i + 1
        if start < seq_len:
            seg_rep = outputs[start:seq_len].mean(dim=0)
            seg_reps.append(seg_rep)
        if seg_reps:
            segments.append(torch.stack(seg_reps, dim=0))
        else:
            segments.append(outputs.mean(dim=0, keepdim=True))

    max_segs = max(seg.shape[0] for seg in segments)
    seg_tensor = torch.zeros(batch_size, max_segs, embed_dim, device=encoder_outputs.device)
    for i, seg in enumerate(segments):
        seg_tensor[i, :seg.shape[0], :] = seg
    return seg_tensor
