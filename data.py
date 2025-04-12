"""
data.py

This module loads glossing data from files in the custom format:
    \t <source sentence>
    \g <gloss>
    \l <translation>

Each sample is separated by a blank line. This module provides a
PyTorch LightningDataModule to load training, validation, and test data.
The raw data is loaded as lists of tokens, and then converted to padded tensors
via a custom collate function.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Dict, Optional
from itertools import chain
from pytorch_lightning import LightningDataModule
from torchtext.vocab import build_vocab_from_iterator
from functools import partial

# Special tokens for our tokenizers.
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


def read_glossing_file_custom(file_path: str) -> Dict[str, List]:
    """
    Reads a glossing file with the following format:
      - Lines starting with "\t" contain the source sentence.
      - Lines starting with "\g" contain the gloss.
      - Lines starting with "\l" contain the translation.
    Each sample is separated by a blank line.

    Args:
        file_path (str): Path to the data file.

    Returns:
        A dictionary with keys:
            "sources": List[List[str]]
            "targets": List[List[str]]
            "translations": List[List[str]]
    """
    samples = []
    current_sample = {"source": None, "gloss": None, "translation": None}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Blank line indicates end of current sample.
            if not line:
                if any(current_sample.values()):
                    samples.append(current_sample)
                current_sample = {"source": None, "gloss": None, "translation": None}
                continue
            if line.startswith("\\t"):
                current_sample["source"] = line[2:].strip().split()
            elif line.startswith("\\g"):
                current_sample["gloss"] = line[2:].strip().split()
            elif line.startswith("\\l"):
                current_sample["translation"] = line[2:].strip().split()
            else:
                continue
        if any(current_sample.values()):
            samples.append(current_sample)
    sources = [s["source"] for s in samples]
    targets = [s["gloss"] for s in samples]
    translations = [s["translation"] for s in samples]
    return {"sources": sources, "targets": targets, "translations": translations}


class GlossingFileData:
    def __init__(self, sources: List[List[str]], targets: List[List[str]], translations: List[List[str]]):
        self.sources = sources
        self.targets = targets
        self.translations = translations


class SequencePairDataset(Dataset):
    def __init__(self, data: GlossingFileData):
        super().__init__()
        self.data = data
        self._length = len(self.data.sources)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, List[str]]:
        return {
            "source": self.data.sources[idx],
            "target": self.data.targets[idx],
            "translation": self.data.translations[idx]
        }


def collate_fn(batch: List[Dict[str, List[str]]],
               source_tokenizer, target_tokenizer, trans_tokenizer,
               max_src_len: int, max_tgt_len: int, max_trans_len: int) -> (
torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Collate function that converts a list of sample dictionaries (with raw token lists)
    into padded tensors.
    Returns:
        src: Tensor of shape (batch, max_src_len)
        src_lengths: Tensor of shape (batch,)
        gloss: Tensor of shape (batch, max_tgt_len)
        trans: Tensor of shape (batch, max_trans_len)
    """
    src_list, src_len_list, gloss_list, trans_list = [], [], [], []
    for sample in batch:
        # Convert source tokens to indices.
        src_tokens = sample["source"]
        src_indices = [source_tokenizer[token] for token in src_tokens]
        src_len_list.append(len(src_indices))
        if len(src_indices) < max_src_len:
            src_indices = src_indices + [source_tokenizer["<pad>"]] * (max_src_len - len(src_indices))
        else:
            src_indices = src_indices[:max_src_len]
        src_list.append(torch.tensor(src_indices, dtype=torch.long))

        # For target (gloss), add start and end tokens.
        tgt_tokens = ["<s>"] + sample["target"] + ["</s>"]
        tgt_indices = [target_tokenizer[token] for token in tgt_tokens]
        if len(tgt_indices) < max_tgt_len:
            tgt_indices = tgt_indices + [target_tokenizer["<pad>"]] * (max_tgt_len - len(tgt_indices))
        else:
            tgt_indices = tgt_indices[:max_tgt_len]
        gloss_list.append(torch.tensor(tgt_indices, dtype=torch.long))

        # For translation.
        trans_tokens = sample["translation"]
        trans_indices = [trans_tokenizer[token] for token in trans_tokens]
        if len(trans_indices) < max_trans_len:
            trans_indices = trans_indices + [trans_tokenizer["<pad>"]] * (max_trans_len - len(trans_indices))
        else:
            trans_indices = trans_indices[:max_trans_len]
        trans_list.append(torch.tensor(trans_indices, dtype=torch.long))

    src_tensor = torch.stack(src_list, dim=0)
    src_lengths = torch.tensor(src_len_list, dtype=torch.long)
    gloss_tensor = torch.stack(gloss_list, dim=0)
    trans_tensor = torch.stack(trans_list, dim=0)
    return src_tensor, src_lengths, gloss_tensor, trans_tensor


def get_collate_fn(source_tokenizer, target_tokenizer, trans_tokenizer,
                   max_src_len: int, max_tgt_len: int, max_trans_len: int):
    """
    Returns a collate function with bound tokenizers and maximum lengths.
    """
    return partial(collate_fn,
                   source_tokenizer=source_tokenizer,
                   target_tokenizer=target_tokenizer,
                   trans_tokenizer=trans_tokenizer,
                   max_src_len=max_src_len,
                   max_tgt_len=max_tgt_len,
                   max_trans_len=max_trans_len)


class GlossingDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for glossing data.
    Expects three files: train, validation, and test.
    """

    def __init__(self, train_file: str, val_file: str, test_file: str, batch_size: int = 32,
                 max_src_len: int = 100, max_tgt_len: int = 30, max_trans_len: int = 50):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.max_trans_len = max_trans_len

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_data_dict = read_glossing_file_custom(self.train_file)
            val_data_dict = read_glossing_file_custom(self.val_file)
            self.train_dataset = SequencePairDataset(GlossingFileData(
                sources=train_data_dict["sources"],
                targets=train_data_dict["targets"],
                translations=train_data_dict["translations"]
            ))
            self.val_dataset = SequencePairDataset(GlossingFileData(
                sources=val_data_dict["sources"],
                targets=val_data_dict["targets"],
                translations=val_data_dict["translations"]
            ))
            # Build vocabularies from training data.
            source_tokens = list(set(token for sentence in train_data_dict["sources"] for token in sentence))
            target_tokens = list(set(token for sentence in train_data_dict["targets"] for token in sentence))
            trans_tokens = list(set(token for sentence in train_data_dict["translations"] for token in sentence))
            for token in SPECIAL_TOKENS:
                if token not in source_tokens:
                    source_tokens.append(token)
                if token not in target_tokens:
                    target_tokens.append(token)
                if token not in trans_tokens:
                    trans_tokens.append(token)
            self.source_alphabet = sorted(source_tokens)
            self.target_alphabet = sorted(target_tokens)
            self.trans_alphabet = sorted(trans_tokens)
            self.source_alphabet_size = len(self.source_alphabet)
            self.target_alphabet_size = len(self.target_alphabet)
            self.trans_alphabet_size = len(self.trans_alphabet)
            # Build tokenizers using torchtext's build_vocab_from_iterator.
            self.source_tokenizer = build_vocab_from_iterator([[token] for token in self.source_alphabet],
                                                              specials=SPECIAL_TOKENS)
            self.target_tokenizer = build_vocab_from_iterator([[token] for token in self.target_alphabet],
                                                              specials=SPECIAL_TOKENS)
            self.trans_tokenizer = build_vocab_from_iterator([[token] for token in self.trans_alphabet],
                                                             specials=SPECIAL_TOKENS)
            self.source_tokenizer.set_default_index(self.source_tokenizer["<unk>"])
            self.target_tokenizer.set_default_index(self.target_tokenizer["<unk>"])
            self.trans_tokenizer.set_default_index(self.trans_tokenizer["<unk>"])
            # Get a bound collate function.
            self._batch_collate = get_collate_fn(self.source_tokenizer, self.target_tokenizer,
                                                 self.trans_tokenizer, self.max_src_len,
                                                 self.max_tgt_len, self.max_trans_len)
        if stage == "test" or stage is None:
            test_data_dict = read_glossing_file_custom(self.test_file)
            self.test_dataset = SequencePairDataset(GlossingFileData(
                sources=test_data_dict["sources"],
                targets=test_data_dict["targets"],
                translations=test_data_dict["translations"]
            ))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self._batch_collate, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self._batch_collate, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self._batch_collate, num_workers=4, persistent_workers=True)


if __name__ == "__main__":
    train_file = "data/Gitksan/git-train-track1-uncovered"
    val_file = "data/Gitksan/git-dev-track1-uncovered"
    test_file = "data/Gitksan/git-test-track1-uncovered"

    dm = GlossingDataModule(train_file, val_file, test_file, batch_size=2)
    dm.setup(stage="fit")
    dm.setup(stage="test")
    print("Training samples:", len(dm.train_dataset))
    print("Validation samples:", len(dm.val_dataset))
    print("Test samples:", len(dm.test_dataset))

    # Print a single batch sample for training (in text format).
    train_loader = dm.test_dataloader()
    for batch in train_loader:
        src_tensor, src_lengths, gloss_tensor, trans_tensor = batch
        print("Batch sample (in text format):")
        for i in range(src_tensor.size(0)):
            src_text = " ".join([dm.source_tokenizer.get_itos()[idx] for idx in src_tensor[i].tolist() if
                                 idx != dm.source_tokenizer["<pad>"]])
            gloss_text = " ".join([dm.target_tokenizer.get_itos()[idx] for idx in gloss_tensor[i].tolist() if
                                   idx != dm.target_tokenizer["<pad>"]])
            trans_text = " ".join([dm.trans_tokenizer.get_itos()[idx] for idx in trans_tensor[i].tolist() if
                                   idx != dm.trans_tokenizer["<pad>"]])
            print(f"Sample {i + 1}:")
            print("  Source:", src_text)
            print("  Gloss: ", gloss_text)
            print("  Trans: ", trans_text)
        break