import json
import os
from typing import List, Optional

import datasets
import pandas as pd
import torch
from datasets import DatasetDict
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

TOTAL_RATIO = 0.05
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.05
TOKENIZATION_BATCH_SIZE = 64
MAX_LENGTH_TEXT = 1024
MAX_LENGTH = 256


class TinyStories(LightningDataModule):
    _tokenizer: PreTrainedTokenizerFast
    _vocab_size: int

    _data_dir: str
    _raw_dir: str
    _processed_dir: str

    _device: torch.device

    _train_texts: Optional[List[str]]
    _validation_texts: Optional[List[str]]
    _test_texts: Optional[List[str]]
    _train_tokens: Optional[Tensor]
    _validation_tokens: Optional[Tensor]
    _test_tokens: Optional[Tensor]

    def __init__(self, data_dir: str, device: torch.device) -> None:
        super().__init__()

        self._tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")  # type: ignore
        self._tokenizer.pad_token = self.tokenizer.eos_token
        self._tokenizer.pad_token = self.tokenizer.eos_token
        self._vocab_size = self.tokenizer.vocab_size

        self._data_dir = data_dir
        self._raw_dir = os.path.join(data_dir, "raw")
        if not os.path.exists(self._raw_dir):
            raise Exception(
                "Raw data directory not found. Please place the raw data in '[data_dir]/raw'. \
                    You can use the 'fetch_raw_data.py' script to download the data."
            )
        self._processed_dir = os.path.join(data_dir, "processed")
        if not os.path.exists(self._processed_dir):
            os.makedirs(self._processed_dir)
        self._device = device

        self._train_texts: Optional[List[str]] = None
        self._validation_texts: Optional[List[str]] = None
        self._test_texts: Optional[List[str]] = None
        self._train_tokens: Optional[Tensor] = None
        self._validation_tokens: Optional[Tensor] = None
        self._test_tokens: Optional[Tensor] = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def prepare_data(self) -> None:
        print("Preparing data...")
        # Load data from disk.
        raw_data_path = os.path.join(self._raw_dir, "data.hf")
        if not os.path.exists(raw_data_path):
            raise Exception("Raw data not found. Please place it in '[data_dir]/raw'.")
        raw_data: DatasetDict = datasets.load_from_disk(raw_data_path)  # type: ignore
        # Convert dataset splits to pandas dataframes.
        train_texts = raw_data["train"].data.to_pandas()
        validation_texts = raw_data["validation"].data.to_pandas()
        # Concatenate the dataframes.
        data = pd.concat([train_texts, validation_texts])

        # Shuffle the data and reset the index.
        short = data[data["text"].apply(len) <= MAX_LENGTH_TEXT]
        short_ratio = len(short) / len(data)
        shuffled_data = short.sample(frac=TOTAL_RATIO / short_ratio, random_state=42).reset_index(drop=True)

        strings = shuffled_data["text"].tolist()

        tokenized_strings = self.tokenizer(
            strings,
            add_special_tokens=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        tokens: Tensor = tokenized_strings["input_ids"].to(torch.int32)  # type: ignore

        # Split the data into train, validation and test sets.
        test_end_index = int(len(shuffled_data) * TEST_RATIO)
        validation_end_index = int(len(shuffled_data) * (TEST_RATIO + VALIDATION_RATIO))

        test_texts = shuffled_data[0:test_end_index]
        validation_texts = shuffled_data[test_end_index:validation_end_index]
        train_texts = shuffled_data[validation_end_index:]

        test_tokens = tokens[0:test_end_index]
        validation_tokens = tokens[test_end_index:validation_end_index]
        train_tokens = tokens[validation_end_index:]

        # Save the data to disk.
        os.makedirs(self._processed_dir, exist_ok=True)
        train_texts.to_pickle(os.path.join(self._processed_dir, "train_texts.pkl"))
        validation_texts.to_pickle(os.path.join(self._processed_dir, "validation_texts.pkl"))
        test_texts.to_pickle(os.path.join(self._processed_dir, "test_texts.pkl"))

        torch.save(train_tokens.clone(), os.path.join(self._processed_dir, "train_tokens.pt"))
        torch.save(validation_tokens.clone(), os.path.join(self._processed_dir, "validation_tokens.pt"))
        torch.save(test_tokens.clone(), os.path.join(self._processed_dir, "test_tokens.pt"))

        info = {
            "vocab_size": self.tokenizer.vocab_size,
            "total_ratio": TOTAL_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "max_length_text": MAX_LENGTH_TEXT,
            "max_length": MAX_LENGTH,
        }
        json.dump(info, open(os.path.join(self._processed_dir, "info.json"), "w"))
        print("Data prepared.")

    def setup(self, stage: str) -> None:
        print("Loading data from disk...")
        info = json.load(open(os.path.join(self._processed_dir, "info.json"), "r"))
        self.vocab_size = info["vocab_size"]

        # Load data from disk
        self.train_texts = pd.read_pickle(os.path.join(self._processed_dir, "train_texts.pkl"))["text"].tolist()
        self.validation_texts = pd.read_pickle(os.path.join(self._processed_dir, "validation_texts.pkl"))[
            "text"
        ].tolist()
        self.test_texts = pd.read_pickle(os.path.join(self._processed_dir, "test_texts.pkl"))["text"].tolist()

        self.train_tokens = torch.load(os.path.join(self._processed_dir, "train_tokens.pt")).long()
        self.validation_tokens = torch.load(os.path.join(self._processed_dir, "validation_tokens.pt")).long()
        self.test_tokens = torch.load(os.path.join(self._processed_dir, "test_tokens.pt")).long()

        self.train_set = TensorDataset(self.train_tokens)
        self.validation_set = TensorDataset(self.validation_tokens)
        self.test_set = TensorDataset(self.test_tokens)
        print("Data loaded.")

    def train_dataloader(self) -> DataLoader[List[Tensor]]:
        if self.train_texts is None or self.train_tokens is None:
            raise Exception("Please call setup() before using this dataloader.")
        pin_memory = self._device != torch.device("cpu")
        if pin_memory:
            train_loader: DataLoader[List[Tensor]] = DataLoader(  # type: ignore
                self.train_set,
                batch_size=4,
                shuffle=True,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device=str(self._device),
            )
        else:
            train_loader = DataLoader(  # type: ignore
                self.train_set,
                batch_size=4,
                shuffle=True,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
            )

        return train_loader

    def val_dataloader(self) -> DataLoader[List[Tensor]]:
        if self.validation_set is None or self.validation_tokens is None:
            raise Exception("Please call setup() before using this dataloader.")
        pin_memory = self._device != torch.device("cpu")
        if pin_memory:
            validation_loader: DataLoader[List[Tensor]] = DataLoader(  # type: ignore
                self.validation_set,
                batch_size=64,
                num_workers=torch.get_num_threads(),
                pin_memory=True,
                pin_memory_device=str(self._device),
            )
        else:
            validation_loader = DataLoader(  # type: ignore
                self.validation_set,
                batch_size=64,
                num_workers=torch.get_num_threads(),
            )
        return validation_loader

    def test_dataloader(self) -> DataLoader[List[Tensor]]:
        if self.test_set is None or self.test_tokens is None:
            raise Exception("Please call setup() before using this dataloader.")
        pin_memory = self._device != torch.device("cpu")
        if pin_memory:
            test_loader: DataLoader[List[Tensor]] = DataLoader(  # type: ignore
                self.test_set,
                batch_size=64,
                num_workers=torch.get_num_threads(),
                pin_memory=True,
                pin_memory_device=str(self._device),
            )
        else:
            test_loader = DataLoader(  # type: ignore
                self.test_set,
                batch_size=64,
                num_workers=torch.get_num_threads(),
            )
        return test_loader
