"""Training Script"""


import torch
import torch.nn.functional as F
from collections import Counter
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from GlossingModel import GlossingPipeline
import pytorch_lightning as pl


#########################################
# 1. Custom Dataset for CSV Data
#########################################
class GlossingDataset(Dataset):
    def __init__(self, csv_file, max_src_len=50, max_tgt_len=50, max_trans_len=50):
        self.data = pd.read_csv(csv_file).dropna().reset_index(drop=True)  # Remove empty rows

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.max_trans_len = max_trans_len

        # Build vocabularies dynamically
        self.src_vocab = self.build_vocab(self.data["Language"], char_level=True)
        self.gloss_vocab = self.build_vocab(self.data["Gloss"], char_level=False)
        self.trans_vocab = self.build_vocab(self.data["Translation"], char_level=False)

    def build_vocab(self, data, char_level=False):
        counter = Counter()
        for item in data.dropna():
            tokens = list(item) if char_level else item.split()
            counter.update(tokens)
        vocab = {tok: i for i, tok in enumerate(counter.keys(), start=2)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        return vocab

    def text_to_tensor(self, text, vocab, char_level=False):
        tokens = list(text) if char_level else text.split()
        indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
        return torch.tensor(indices, dtype=torch.long)

    def tensor_to_text(self, tensor, vocab):
        inv_vocab = {idx: tok for tok, idx in vocab.items()}
        return " ".join([inv_vocab.get(idx, "<unk>") for idx in tensor.tolist()])

    def tokenize(self, text, vocab, max_len, char_level=False):
        tokens = list(text) if char_level else text.split()
        indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
        indices += [vocab["<pad>"]] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = str(row["Language"])
        gloss_text = str(row["Gloss"])
        trans_text = str(row["Translation"])

        # Tokenize source and compute its original length (capped by max_src_len)
        src_indices = self.tokenize(src_text, self.src_vocab, self.max_src_len, char_level=True)
        # For char-level, use the number of characters (or max_src_len if longer)
        src_len = min(len(list(src_text)), self.max_src_len)

        gloss_indices = self.tokenize(gloss_text, self.gloss_vocab, self.max_tgt_len)
        trans_indices = self.tokenize(trans_text, self.trans_vocab, self.max_trans_len)

        return (torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(src_len, dtype=torch.long),
                torch.tensor(gloss_indices, dtype=torch.long),
                torch.tensor(trans_indices, dtype=torch.long))


def collate_fn(batch):
    src_batch, src_len_batch, tgt_batch, trans_batch = zip(*batch)
    return (torch.stack(src_batch, dim=0),
            torch.stack(src_len_batch, dim=0),
            torch.stack(tgt_batch, dim=0),
            torch.stack(trans_batch, dim=0))


#########################################
# 2. Main Training Script using PyTorch Lightning
#########################################
if __name__ == '__main__':
    pl.seed_everything(42)
    # Dummy vocabulary dictionaries for demonstration.
    src_vocab = {ch: i for i, ch in enumerate(list("abcdefghijklmnopqrstuvwxyz-"))}
    src_vocab["<pad>"] = len(src_vocab)
    src_vocab["<unk>"] = len(src_vocab)

    gloss_vocab = {tok: i for i, tok in enumerate(["shortage-FEM.NOM.SG", "wine-NEUT.GEN.SG"])}
    gloss_vocab["<pad>"] = len(gloss_vocab)
    gloss_vocab["<unk>"] = len(gloss_vocab)

    trans_vocab = {tok: i for i, tok in enumerate(["a", "wine", "shortage"])}
    trans_vocab["<pad>"] = len(trans_vocab)
    trans_vocab["<unk>"] = len(trans_vocab)

    # Create dataset and dataloader.
    dataset = GlossingDataset("data/Dummy_Dataset.csv", max_src_len=50, max_tgt_len=20, max_trans_len=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn,
                              num_workers=6, persistent_workers=True)

    # Hyperparameters.
    char_vocab_size = len(dataset.src_vocab)
    gloss_vocab_size = len(dataset.gloss_vocab)
    trans_vocab_size = len(dataset.trans_vocab)
    embed_dim = 256  # Must be divisible by num_heads.
    num_heads = 8
    ff_dim = 512
    num_layers = 6
    dropout = 0.1
    use_gumbel = True
    learning_rate = 0.001
    gloss_pad_idx = dataset.gloss_vocab["<pad>"]

    # Initialize the integrated LightningModule.
    model = GlossingPipeline(char_vocab_size, gloss_vocab_size, trans_vocab_size,
                             embed_dim, num_heads, ff_dim, num_layers, dropout, use_gumbel,
                             learning_rate, gloss_pad_idx)

    # Initialize PyTorch Lightning trainer.
    trainer = pl.Trainer(max_epochs=3,
                         accelerator="auto",
                         log_every_n_steps=10,
                         enable_progress_bar=True)
    trainer.fit(model, dataloader)

    # Saving the trained model here
    trainer.save_checkpoint("glossing_model.ckpt")


