"""Training Script"""
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from collections import Counter
from GlossingModel import GlossingPipeline

#########################################
# 1. Custom Dataset for Glossing
#########################################
class GlossingDataset(Dataset):
    def __init__(self, csv_file, max_src_len=20, max_tgt_len=20, max_trans_len=20):
        self.data = pd.read_csv(csv_file).dropna().reset_index(drop=True)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.max_trans_len = max_trans_len

        # Build vocabularies dynamically.
        self.src_vocab = self.build_vocab(self.data["Language"], char_level=True)
        self.gloss_vocab = self.build_vocab(self.data["Gloss"], char_level=False)
        self.trans_vocab = self.build_vocab(self.data["Translation"], char_level=False)

        # Ensure special tokens exist for gloss.
        for token in ["<s>", "</s>", "<pad>", "<unk>"]:
            if token not in self.gloss_vocab:
                self.gloss_vocab[token] = len(self.gloss_vocab)

    def build_vocab(self, data, char_level=False):
        counter = Counter()
        for item in data.dropna():
            tokens = list(item) if char_level else item.split()
            counter.update(tokens)
        # Reserve 0 for <pad> and 1 for <unk>.
        vocab = {tok: i for i, tok in enumerate(counter.keys(), start=2)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        return vocab

    def text_to_tensor(self, text, vocab, max_len, char_level=False):
        tokens = list(text) if char_level else text.split()
        indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
        indices += [vocab["<pad>"]] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def tensor_to_text(self, tensor, vocab):
        inv_vocab = {idx: tok for tok, idx in vocab.items()}
        tokens = [inv_vocab.get(idx.item(), "<unk>") for idx in tensor if idx.item() != self.gloss_vocab["<pad>"]]
        return " ".join(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = str(row["Language"])
        gloss_text = str(row["Gloss"])
        trans_text = str(row["Translation"])

        src_tensor = self.text_to_tensor(src_text, self.src_vocab, self.max_src_len, char_level=True)
        # For gloss, add start and end tokens.
        gloss_tokens = ["<s>"] + gloss_text.split() + ["</s>"]
        gloss_indices = [self.gloss_vocab.get(tok, self.gloss_vocab["<unk>"]) for tok in gloss_tokens]
        # Truncate and pad gloss to fixed length.
        gloss_indices = gloss_indices[:self.max_tgt_len]
        if len(gloss_indices) < self.max_tgt_len:
            gloss_indices += [self.gloss_vocab["<pad>"]] * (self.max_tgt_len - len(gloss_indices))
        gloss_tensor = torch.tensor(gloss_indices, dtype=torch.long)
        trans_tensor = self.text_to_tensor(trans_text, self.trans_vocab, self.max_trans_len, char_level=False)

        src_len = min(len(list(src_text)), self.max_src_len)
        return src_tensor, src_len, gloss_tensor, trans_tensor

def collate_fn(batch):
    src_list, src_len_list, tgt_list, trans_list = zip(*batch)
    src = torch.stack(src_list, dim=0)
    src_len = torch.tensor(src_len_list, dtype=torch.long)
    tgt = torch.stack(tgt_list, dim=0)
    trans = torch.stack(trans_list, dim=0)
    return src, src_len, tgt, trans

#########################################
# 2. Main Training Script using PyTorch Lightning
#########################################
if __name__ == '__main__':
    pl.seed_everything(42, workers=True)

    # Create dataset and dataloader.
    dataset = GlossingDataset("data/Dummy_Dataset.csv", max_src_len=20, max_tgt_len=20, max_trans_len=20)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=collate_fn,
                            num_workers=17, persistent_workers=True)

    # Hyperparameters.
    char_vocab_size = len(dataset.src_vocab)
    gloss_vocab_size = len(dataset.gloss_vocab)
    trans_vocab_size = len(dataset.trans_vocab)
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_layers = 6
    dropout = 0.1
    use_gumbel = True
    learning_rate = 0.001
    gloss_pad_idx = dataset.gloss_vocab["<pad>"]

    # Instantiate the integrated model.
    model = GlossingPipeline(
        char_vocab_size,
        gloss_vocab_size,
        trans_vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        dropout,
        use_gumbel,
        learning_rate,
        gloss_pad_idx
    )

    # Initialize trainer and train.
    trainer = pl.Trainer(max_epochs=25, accelerator="auto", log_every_n_steps=1,
                         deterministic=True)
    trainer.fit(model, dataloader)

    # Save the trained model checkpoint.
    checkpoint_path = "glossing_model.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")