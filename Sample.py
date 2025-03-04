import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from GlossingModel import GlossingPipeline
from main import GlossingDataset, collate_fn  # Assuming your dataset and collate_fn are defined in main.py


def main():
    pl.seed_everything(42, workers=True)

    # Load the dataset.
    dataset = GlossingDataset("data/Dummy_Dataset.csv", max_src_len=100, max_tgt_len=50, max_trans_len=50)

    # For a sample prediction, we'll take an example from the dataset
    src_tensor, src_len, gloss_tensor, trans_tensor = dataset[20]

    # Convert source indices to one-hot vectors.
    char_vocab_size = len(dataset.src_vocab)
    src_features = F.one_hot(src_tensor, num_classes=char_vocab_size).float().unsqueeze(
        0)  # (1, src_seq_len, char_vocab_size)
    src_len_tensor = torch.tensor([src_len], dtype=torch.long)  # (1,)

    # Add batch dimension to target and translation.
    tgt_tensor = gloss_tensor.unsqueeze(0)  # (1, tgt_seq_len)
    trans_tensor = trans_tensor.unsqueeze(0)  # (1, trans_seq_len)

    # Define hyperparameters matching your training configuration.
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_layers = 6
    dropout = 0.1
    use_gumbel = True
    learning_rate = 0.001
    use_relative = True
    max_relative_position = 64
    gloss_pad_idx = dataset.gloss_vocab["<pad>"]

    # Load the trained model checkpoint.
    checkpoint_path = "glossing_model.ckpt"  # Ensure this checkpoint exists
    model = GlossingPipeline.load_from_checkpoint(
        checkpoint_path,
        char_vocab_size=char_vocab_size,
        gloss_vocab_size=len(dataset.gloss_vocab),
        trans_vocab_size=len(dataset.trans_vocab),
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_gumbel=use_gumbel,
        learning_rate=learning_rate,
        gloss_pad_idx=gloss_pad_idx,
        use_relative=use_relative,
        max_relative_position=max_relative_position
    )
    model.eval()

    # Run prediction.
    with torch.no_grad():
        logits, morpheme_count, tau, seg_probs = model(src_features, src_len_tensor, tgt_tensor, trans_tensor,
                                                       learn_segmentation=True)

    # Get predicted gloss tokens (remove batch dimension).
    predicted_indices = torch.argmax(logits, dim=-1).squeeze(0)

    # Create an inverse gloss vocabulary for decoding.
    inv_gloss_vocab = {idx: token for token, idx in dataset.gloss_vocab.items()}
    predicted_gloss = " ".join([inv_gloss_vocab.get(idx.item(), "<unk>") for idx in predicted_indices])

    # stop decoding when we see the stop token
    stop_index = predicted_gloss.index('</s>')


    # Print the sample input and predicted gloss.
    print("Sample Input (Language):")
    print(dataset.tensor_to_text(src_tensor, dataset.src_vocab))
    print("\nPredicted Gloss:")
    print(predicted_gloss[:stop_index + 4])


if __name__ == "__main__":
    main()