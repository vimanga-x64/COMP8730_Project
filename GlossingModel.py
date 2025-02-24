"""Decoder/Glossing Model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR


from Encoder import TransformerCharEncoder
from MorphemeSegmenter import MorphemeSegmenter
from Utilities import max_pool_2d, make_mask_2d

class MorphemeGlossingModel(LightningModule):
    def __init__(
        self,
        source_alphabet_size: int,
        target_alphabet_size: int,
        hidden_size: int = 256,
        num_encoder_layers: int = 6,
        dropout: float = 0.1,
        scheduler_gamma: float = 1.0,
        learn_segmentation: bool = True,
        classify_num_morphemes: bool = False,
    ):
        """
        Our model encodes a source word (as a sequence of characters) using a Transformer-based encoder.
        Then, if learn_segmentation is enabled (for Track 1), it learns to segment the word into morphemes
        using our improved unsupervised morpheme segmenter. Finally, a linear classifier predicts gloss labels
        from the morpheme representations.
        """
        super().__init__()
        self.source_alphabet_size = source_alphabet_size
        self.target_alphabet_size = target_alphabet_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.scheduler_gamma = scheduler_gamma
        self.learn_segmentation = learn_segmentation
        self.classify_num_morphemes = classify_num_morphemes

        self.save_hyperparameters()

        # Source character embeddings and Transformer encoder.
        self.embeddings = nn.Embedding(
            num_embeddings=self.source_alphabet_size,
            embedding_dim=self.hidden_size,
            padding_idx=0,
        )
        self.encoder = TransformerCharEncoder(
            vocab_size=self.source_alphabet_size,
            d_model=self.hidden_size,
            nhead=8,
            num_layers=num_encoder_layers,
            dim_feedforward=512,
            dropout=self.dropout,
            max_len=512,
        )

        # Gloss classifier: maps morpheme representations to gloss token probabilities.
        self.classifier = nn.Linear(self.hidden_size, self.target_alphabet_size)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

        # Unsupervised segmentation (learned segmentation for Track 1)
        if self.learn_segmentation:
            self.segmenter = MorphemeSegmenter(self.hidden_size)

        # Optionally, a classifier to predict number of morphemes.
        if self.classify_num_morphemes:
            self.num_morpheme_classifier = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, 10),
            )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.001, weight_decay=0.0)
        scheduler = ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]

    def encode_sentences(self, sentences: torch.Tensor, sentence_lengths: torch.Tensor) -> torch.Tensor:
        # sentences: (batch, seq_len) with character indices.
        embeddings = self.embeddings(sentences)  # (batch, seq_len, hidden_size)
        # Pass through Transformer encoder.
        encodings = self.encoder(sentences)
        return encodings

    def get_words(self, encodings: torch.Tensor, word_extraction_index: torch.Tensor) -> torch.Tensor:
        # Extract word representations based on word_extraction_index.
        # For simplicity, assume word_extraction_index maps indices from the flattened encoder output.
        encodings = encodings.reshape(-1, self.hidden_size)
        num_words, chars_per_word = word_extraction_index.shape
        word_extraction_index_flat = word_extraction_index.flatten()
        word_encodings = torch.index_select(encodings, dim=0, index=word_extraction_index_flat)
        word_encodings = word_encodings.reshape(num_words, chars_per_word, self.hidden_size)
        return word_encodings

    def get_num_morphemes(self, word_encodings: torch.Tensor, word_lengths: torch.Tensor):
        # Only used if classify_num_morphemes is True.
        word_encodings = max_pool_2d(word_encodings, word_lengths)  # (batch, hidden_size)
        num_morpheme_scores = self.num_morpheme_classifier(word_encodings)  # (batch, 10)
        num_morpheme_predictions = torch.argmax(num_morpheme_scores, dim=-1)
        num_morpheme_predictions = torch.minimum(num_morpheme_predictions, word_lengths)
        num_morpheme_predictions = torch.clamp(num_morpheme_predictions, min=1)
        return {"scores": num_morpheme_scores, "predictions": num_morpheme_predictions}

    def forward(self, batch, training: bool = True):
        """
        batch: an object with at least the following attributes:
          - batch.sentences: Tensor of shape (batch, seq_len) for source characters.
          - batch.sentence_lengths: Tensor of shape (batch,) with valid lengths.
          - batch.word_extraction_index: Tensor mapping positions for word extraction.
          - batch.word_lengths: Tensor with the number of valid characters per word.
          - batch.word_target_lengths: Tensor with the expected number of morphemes per word.
        """
        # Encode the sentences using our Transformer encoder.
        char_encodings = self.encode_sentences(batch.sentences, batch.sentence_lengths)
        # Extract word-level representations (for our purposes, assume each sentence is a word).
        word_encodings = self.get_words(char_encodings, batch.word_extraction_index)

        if self.classify_num_morphemes:
            num_morphemes_info = self.get_num_morphemes(word_encodings, batch.word_lengths)
            if training:
                num_morphemes = batch.word_target_lengths
            else:
                num_morphemes = num_morphemes_info["predictions"]
        else:
            num_morphemes = batch.word_target_lengths

        if self.learn_segmentation:
            # Learn segmentation from word_encodings (unsupervised segmentation for Track 1)
            morpheme_encodings, best_path_matrix = self.segmenter(
                word_encodings,
                batch.word_lengths,
                num_morphemes,
                training=training
            )
        else:
            # If segmentation is provided, extract morphemes accordingly.
            # (For Track 2, for example; not implemented here.)
            morpheme_encodings = word_encodings
            best_path_matrix = None

        # Gloss prediction: apply linear classifier to each morpheme representation.
        morpheme_scores = self.classifier(morpheme_encodings)  # (batch, num_morphemes, target_vocab_size)

        return {
            "morpheme_scores": morpheme_scores,    # Predictions for gloss labels.
            "best_path_matrix": best_path_matrix,   # Learned segmentation details.
        }

# Example dummy Batch class for testing.
class DummyBatch:
    def __init__(self, batch_size, src_seq_len, word_len, num_words):
        self.sentences = torch.randint(1, 100, (batch_size, src_seq_len))
        self.sentence_lengths = torch.full((batch_size,), src_seq_len, dtype=torch.long)
        # For simplicity, assume each sentence is a word; word_extraction_index selects all positions.
        self.word_extraction_index = torch.arange(src_seq_len).unsqueeze(0).expand(batch_size, src_seq_len)
        self.word_lengths = torch.full((batch_size,), word_len, dtype=torch.long)
        # Expected number of morphemes per word (gold segmentation for Track 1)
        self.word_target_lengths = torch.randint(1, 4, (batch_size,))

if __name__ == "__main__":
    # Parameters for dummy data.
    batch_size = 4
    src_seq_len = 20
    word_len = 20
    num_words = batch_size
    source_alphabet_size = 100
    target_alphabet_size = 50
    hidden_size = 256

    # Create a dummy batch.
    batch = DummyBatch(batch_size, src_seq_len, word_len, num_words)

    # Instantiate our MorphemeGlossingModel.
    model = MorphemeGlossingModel(
        source_alphabet_size=source_alphabet_size,
        target_alphabet_size=target_alphabet_size,
        hidden_size=hidden_size,
        num_encoder_layers=2,  # Use fewer layers for testing.
        dropout=0.1,
        scheduler_gamma=1.0,
        learn_segmentation=True,
        classify_num_morphemes=False  # For now, we don't need num_morpheme classification.
    )

    # Run a forward pass in training mode.
    model.train()
    outputs = model(batch, training=True)
    print("Morpheme scores shape (training):", outputs["morpheme_scores"].shape)
    if outputs["best_path_matrix"] is not None:
        print("Best path matrix shape (training):", outputs["best_path_matrix"].shape)

    # Run a forward pass in inference mode.
    model.eval()
    with torch.no_grad():
        outputs = model(batch, training=False)
    print("Morpheme scores shape (inference):", outputs["morpheme_scores"].shape)
    if outputs["best_path_matrix"] is not None:
        print("Best path matrix shape (inference):", outputs["best_path_matrix"].shape)