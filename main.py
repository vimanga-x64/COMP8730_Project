"""Training Script"""


import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from GlossingModel import GlossingPipeline
import pytorch_lightning as pl


#########################################
# 1. Custom Dataset for CSV Data i.e. Dummy_Dataset.csv
#########################################
class GlossingDataset(Dataset):
    """
    A dataset that loads a CSV file with columns:
      "Language", "Gloss", "Translation"
    and tokenizes them.
    """
    def __init__(self, csv_file, src_vocab, gloss_vocab, trans_vocab,
                 max_src_len: int = 50, max_tgt_len: int = 50, max_trans_len: int = 50):
        self.data = pd.read_csv(csv_file)
        self.src_vocab = src_vocab      # dictionary: character -> index
        self.gloss_vocab = gloss_vocab  # dictionary: token -> index
        self.trans_vocab = trans_vocab  # dictionary: token -> index
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.max_trans_len = max_trans_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_str = str(row["Language"])       # e.g., "inopi-a"
        gloss_str = str(row["Gloss"])          # e.g., "shortage-FEM.NOM.SG wine-NEUT.GEN.SG"
        trans_str = str(row["Translation"])    # e.g., "a wine shortage"

        # Tokenize source at character level.
        src_indices = [self.src_vocab.get(ch, self.src_vocab.get("<unk>")) for ch in src_str][:self.max_src_len]
        src_len = len(src_indices)
        src_indices = src_indices + [self.src_vocab.get("<pad>")] * (self.max_src_len - src_len)

        # Tokenize gloss (space-separated tokens).
        gloss_tokens = gloss_str.strip().split()
        tgt_indices = [self.gloss_vocab.get(tok, self.gloss_vocab.get("<unk>")) for tok in gloss_tokens][:self.max_tgt_len]
        tgt_indices = tgt_indices + [self.gloss_vocab.get("<pad>")] * (self.max_tgt_len - len(tgt_indices))

        # Tokenize translation (space-separated tokens).
        trans_tokens = trans_str.strip().split()
        trans_indices = [self.trans_vocab.get(tok, self.trans_vocab.get("<unk>")) for tok in trans_tokens][:self.max_trans_len]
        trans_indices = trans_indices + [self.trans_vocab.get("<pad>")] * (self.max_trans_len - len(trans_indices))

        return (torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(src_len, dtype=torch.long),
                torch.tensor(tgt_indices, dtype=torch.long),
                torch.tensor(trans_indices, dtype=torch.long))


def collate_fn(batch):
    src_batch, src_len_batch, tgt_batch, trans_batch = zip(*batch)
    src_batch = torch.stack(src_batch, dim=0)
    src_len_batch = torch.stack(src_len_batch, dim=0)
    tgt_batch = torch.stack(tgt_batch, dim=0)
    trans_batch = torch.stack(trans_batch, dim=0)
    return src_batch, src_len_batch, tgt_batch, trans_batch


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
    dataset = GlossingDataset("data/Dummy_Dataset.csv", src_vocab, gloss_vocab, trans_vocab,
                              max_src_len=50, max_tgt_len=20, max_trans_len=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn,
                              num_workers=6, persistent_workers=True)

    # Hyperparameters.
    char_vocab_size = len(src_vocab)
    gloss_vocab_size = len(gloss_vocab)
    trans_vocab_size = len(trans_vocab)
    embed_dim = 256  # Must be divisible by num_heads.
    num_heads = 8
    ff_dim = 512
    num_layers = 6
    dropout = 0.1
    use_gumbel = True
    learning_rate = 0.001
    gloss_pad_idx = gloss_vocab["<pad>"]

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

