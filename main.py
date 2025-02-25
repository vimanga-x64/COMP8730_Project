"""Training and Evalulation Script"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer

# Import your glossing model from your separate module.
# This assumes your model is defined in a file named GlossingModel.py and has a class called MorphemeGlossingModel.
from GlossingModel import MorphemeGlossingModel


# -----------------------------
# Utilities for Tokenization and Vocabulary
# -----------------------------
def build_char_vocab(strings):
    """Build a vocabulary mapping for characters."""
    vocab = {"<pad>": 0, "<unk>": 1}
    for s in strings:
        for ch in s:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def tokenize_string(s):
    """Tokenize a string into a list of characters."""
    return list(s)


def encode_string(s, vocab):
    """Encode a list of tokens using the vocabulary."""
    return [vocab.get(token, vocab["<unk>"]) for token in s]


def pad_sequence(seq, max_len, pad_value=0):
    """Pad a sequence to max_len with pad_value."""
    return seq + [pad_value] * (max_len - len(seq))


# -----------------------------
# Dataset Class
# -----------------------------
class GlossingDataset(Dataset):
    """
    Dataset for glossing.
    Expects a CSV with columns:
      - Language: Source sentence.
      - Translation: English translation (unused in this script).
      - Gloss: Target gloss (morpheme glosses separated by whitespace).
    """

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # Convert columns to string to avoid type issues.
        self.df["Language"] = self.df["Language"].astype(str)
        self.df["Gloss"] = self.df["Gloss"].astype(str)
        self.df["Translation"] = self.df["Translation"].astype(str)

        # Build vocabularies.
        self.source_vocab = build_char_vocab(self.df["Language"].tolist())
        self.gloss_vocab = build_char_vocab(self.df["Gloss"].tolist())

        # Process each row into a sample.
        self.samples = []
        for idx, row in self.df.iterrows():
            source = row["Language"]
            gloss = row["Gloss"]
            # Tokenize and encode source.
            src_tokens = tokenize_string(source)
            src_encoded = encode_string(src_tokens, self.source_vocab)
            # Assume gloss tokens (morphemes) are separated by whitespace.
            gloss_tokens = gloss.split()
            target_length = len(gloss_tokens)
            # For simplicity, take the first character of each gloss token as the label.
            morpheme_targets = [self.gloss_vocab.get(token[0], self.gloss_vocab["<unk>"]) for token in gloss_tokens]

            sample = {
                "source_encoded": src_encoded,
                "word_target_length": target_length,
                "morpheme_targets": morpheme_targets
            }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -----------------------------
# Collate Function
# -----------------------------
def collate_fn(samples):
    batch_size = len(samples)
    max_src_len = max(len(s["source_encoded"]) for s in samples)
    max_target_len = max(s["word_target_length"] for s in samples)

    # Pad source sequences.
    source_seqs = [pad_sequence(s["source_encoded"], max_src_len, pad_value=0) for s in samples]
    sentence_lengths = [len(s["source_encoded"]) for s in samples]

    # In this simplified case, treat each sentence as one word.
    word_extraction_index = [list(range(max_src_len)) for _ in range(batch_size)]
    word_lengths = sentence_lengths
    word_target_lengths = [s["word_target_length"] for s in samples]

    # Pad morpheme targets.
    morpheme_targets = [pad_sequence(s["morpheme_targets"], max_target_len, pad_value=0) for s in samples]

    # Ensure all tensors are of type torch.long.
    batch = type("Batch", (), {})()  # Create an empty object.
    batch.sentences = torch.tensor(source_seqs, dtype=torch.long)
    batch.sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.long)
    batch.word_extraction_index = torch.tensor(word_extraction_index, dtype=torch.long)
    batch.word_lengths = torch.tensor(word_lengths, dtype=torch.long)
    batch.word_target_lengths = torch.tensor(word_target_lengths, dtype=torch.long)
    batch.morpheme_targets = torch.tensor(morpheme_targets, dtype=torch.long)
    return batch


# -----------------------------
# Main Training Script
# -----------------------------
def main():
    # Load the CSV dataset
    torch.manual_seed(42)
    csv_file = "data/Dummy_Dataset.csv"  # Ensure this file exists.
    dataset = GlossingDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print("Number of samples:", len(dataset))
    print("Source vocab size:", len(dataset.source_vocab))
    print("Gloss vocab size:", len(dataset.gloss_vocab))

    # Instantiate your glossing model.
    model = MorphemeGlossingModel(
        source_alphabet_size=len(dataset.source_vocab),
        target_alphabet_size=len(dataset.gloss_vocab),
        hidden_size=256,
        num_encoder_layers=2,  # For testing.
        dropout=0.1,
        scheduler_gamma=1.0,
        learn_segmentation=True,  # For Track 1.
        classify_num_morphemes=False
    )

    # Train the model for a few epochs.
    trainer = Trainer(max_epochs=3, enable_progress_bar=True)
    trainer.fit(model, dataloader)

    # Evaluate on a sample.
    sample_source = "Akana taa suu."
    sample_translation = "He isn't to go home."  # Unused currently.
    sample_gloss = "he INFL.NEG go home"

    # Process the sample.
    src_tokens = tokenize_string(sample_source)
    src_encoded = encode_string(src_tokens, dataset.source_vocab)
    word_target_length = len(sample_gloss.split())
    gloss_tokens = sample_gloss.split()
    morpheme_targets = [dataset.gloss_vocab.get(token[0], dataset.gloss_vocab["<unk>"]) for token in gloss_tokens]

    sample_batch = type("Batch", (), {})()
    sample_batch.sentences = torch.tensor([pad_sequence(src_encoded, max_len=len(src_encoded))], dtype=torch.long)
    sample_batch.sentence_lengths = torch.tensor([len(src_encoded)], dtype=torch.long)
    sample_batch.word_extraction_index = torch.tensor([list(range(len(src_encoded)))], dtype=torch.long)
    sample_batch.word_lengths = torch.tensor([len(src_encoded)], dtype=torch.long)
    sample_batch.word_target_lengths = torch.tensor([word_target_length], dtype=torch.long)
    sample_batch.morpheme_targets = torch.tensor([morpheme_targets], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        outputs = model(sample_batch, training=False)
    morpheme_scores = outputs["morpheme_scores"]
    best_path_matrix = outputs["best_path_matrix"]

    print("Sample source (encoded):", sample_batch.sentences)
    print("Predicted morpheme scores shape:", morpheme_scores.shape)
    if best_path_matrix is not None:
        print("Predicted best path matrix shape:", best_path_matrix.shape)

    predicted_indices = morpheme_scores.argmax(dim=-1)
    inv_gloss_vocab = {v: k for k, v in dataset.gloss_vocab.items()}
    predicted_gloss = [inv_gloss_vocab.get(idx.item(), "<unk>") for idx in predicted_indices[0]]
    print("Predicted gloss for sample:", " ".join(predicted_gloss))


if __name__ == "__main__":
    main()