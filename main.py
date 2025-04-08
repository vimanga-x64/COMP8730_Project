import pytorch_lightning as pl
from GlossingModel import GlossingPipeline
from data import GlossingDataModule
from metrics import compute_word_level_gloss_accuracy, compute_morpheme_level_gloss_accuracy
import argparse

language_code_mapping = {
    "Arapaho": "arp",
    "Gitksan": "git",
    "Lezgi": "lez",
    "Natugu": "ntu",
    "Nyangbo": "nyb",
    "Tsez": "ddo",
    "Uspanteko": "usp",
}

def make_argument_parser():
    parser = argparse.ArgumentParser(description="Glossing Model Arguments")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=list(language_code_mapping.keys()),
    )

    parser.add_argument("--batch",
                        type=int, default=128, required=False, help="Batch size.")
    parser.add_argument("--layers",
                        type=int, default=2, required=False, help="Number of Layers")
    parser.add_argument("--dropout",
                        type=float, default=0.1, required=False, help="Dropout for each Layer")
    parser.add_argument("--lr",
                        type=float, default=0.001, required=False, help="The learning rate")
    parser.add_argument("--embdim",
                        type=int, default=128, required=False, help="Embedding Dimensions")
    parser.add_argument("--ffdim",
                        type=int, default=512, required=False, help="FeedForward Dimension")
    parser.add_argument("--numheads",
                        type=int, default=16, required=False, help="Number of heads")
    parser.add_argument("--epochs",
                        type=int, default=25, required=False, help="Number of Epochs")

    return parser.parse_args()


if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    args = make_argument_parser()

    language = args.language
    language_code = language_code_mapping[language] # language is the key in the map


    # Define file paths for training, validation, and test data.
    train_file = f"data/{language}/{language_code}-train-track1-uncovered"
    val_file = f"data/{language}/{language_code}-dev-track1-uncovered"
    test_file = f"data/{language}/{language_code}-test-track1-uncovered"

    # Create the DataModule instance.
    dm = GlossingDataModule(train_file=train_file, val_file=val_file, test_file=test_file,
                            batch_size=args.batch)
    dm.setup(stage="fit")
    dm.setup(stage="test")

    # Retrieve vocabulary sizes from the DataModule.
    char_vocab_size = dm.source_alphabet_size   # Source vocabulary size (for characters)
    gloss_vocab_size = dm.target_alphabet_size    # Gloss vocabulary size (for gloss tokens)
    trans_vocab_size = dm.trans_alphabet_size      # Translation vocabulary size

    # Define hyperparameters.
    embed_dim = args.embdim
    num_heads = args.numheads
    ff_dim = args.ffdim
    num_layers = args.layers
    dropout = args.dropout
    use_gumbel = True
    learning_rate = args.lr
    use_relative = True
    max_relative_position = 64
    # the gloss tokenizer uses "<pad>" as the padding token.
    gloss_pad_idx = dm.target_tokenizer.get_stoi()["<pad>"]

    # Instantiate the integrated glossing model.
    model = GlossingPipeline(
        char_vocab_size=char_vocab_size,
        gloss_vocab_size=gloss_vocab_size,
        trans_vocab_size=trans_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_gumbel=use_gumbel,
        learning_rate=learning_rate,
        gloss_pad_idx=gloss_pad_idx,
        use_relative=use_relative,
        max_relative_position=max_relative_position,
    )

    # Configure the PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        log_every_n_steps=5,
        deterministic=True
    )

    # Train the model.
    trainer.fit(model, dm)

    # Save the trained model checkpoint.
    checkpoint_path = f"models/glossing_model_{language_code}.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

  # Get predictions and true glosses.
    print("Loading the model from the checkpoint...")
    model = GlossingPipeline.load_from_checkpoint(checkpoint_path)
    predictions = trainer.predict(model, dataloaders=dm.test_dataloader())

    # Create an inverse mapping for the gloss vocabulary.
    inv_gloss_vocab = {idx: token for token, idx in dm.target_tokenizer.get_stoi().items()}

    # Extract true glosses from the test dataset.
    true_glosses = []
    for batch in dm.test_dataloader():
        _, _, tgt_batch, _ = batch
        for tgt in tgt_batch:
            gloss_tokens = [inv_gloss_vocab.get(idx.item(), "<unk>") for idx in tgt if idx.item() != gloss_pad_idx]
            if "</s>" in gloss_tokens:
                gloss_tokens = gloss_tokens[:gloss_tokens.index("</s>")]
            true_gloss = " ".join(gloss_tokens)
            true_glosses.append(true_gloss)

    # Process and print predictions alongside true glosses.
    predicted_glosses = []
    sample_index = 0  # Global sample index across all batches
    print("\nPredictions and True Glosses:")
    for batch in predictions:
        for pred in batch:
            tokens = [inv_gloss_vocab.get(idx.item(), "<unk>") for idx in pred if idx.item() != gloss_pad_idx]
            if "</s>" in tokens:
                tokens = tokens[:tokens.index("</s>")]
            predicted_gloss = " ".join(tokens)
            predicted_glosses.append(predicted_gloss)
            # Print predicted gloss and true gloss side by side.
            print(f"Sample {sample_index + 1}:")
            print(f"  Predicted Gloss: {predicted_gloss}")
            print(f"  True Gloss:     {true_glosses[sample_index]}")
            print()  # Add a blank line for readability.
            sample_index += 1  # Increment the global sample index

    # Calculate and print word-level and morpheme-level gloss accuracy.
    word_level_accuracy = compute_word_level_gloss_accuracy(predicted_glosses, true_glosses)
    morpheme_level_accuracy = compute_morpheme_level_gloss_accuracy(predicted_glosses, true_glosses)

    print("word level accuracy:", word_level_accuracy)
    print("morpheme level accuracy:", morpheme_level_accuracy)