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

from .constants import PROCESSED_DATA_PATH, RAW_DATA_PATH

TOTAL_RATIO = 0.05
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.05
TOKENIZATION_BATCH_SIZE = 64
MAX_LENGTH_TEXT = 1024
MAX_LENGTH = 256
DATA_LOADER_BATCH_SIZE = 4


class TinyStories(LightningDataModule):
    _tokenizer: PreTrainedTokenizerFast
    _vocab_size: int

    _device: torch.device

    _train_texts: Optional[List[str]]
    _validation_texts: Optional[List[str]]
    _test_texts: Optional[List[str]]
    _train_tokens: Optional[Tensor]
    _validation_tokens: Optional[Tensor]
    _test_tokens: Optional[Tensor]

    def __init__(self, device: torch.device) -> None:
        super().__init__()

        self._tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")  # type: ignore
        self._tokenizer.pad_token = self.tokenizer.eos_token
        self._tokenizer.pad_token = self.tokenizer.eos_token
        self._vocab_size = self.tokenizer.vocab_size

        if not os.path.exists(RAW_DATA_PATH):
            raise Exception(
                f"Raw data directory not found. Please place the raw data in [{RAW_DATA_PATH}] \
                    from 'data/constants.py'. You can use the 'fetch_raw_data.py' script to download the data."
            )

        if not os.path.exists(PROCESSED_DATA_PATH):
            os.makedirs(PROCESSED_DATA_PATH)
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
        if not os.path.exists(RAW_DATA_PATH):
            raise Exception(
                f"Raw data directory not found. Please place the raw data in [{RAW_DATA_PATH}] \
                    from 'data/constants.py'. You can use the 'fetch_raw_data.py' script to download the data."
            )
        # Load data from disk.
        raw_data: DatasetDict = datasets.load_from_disk(RAW_DATA_PATH)  # type: ignore
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
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        train_texts.to_pickle(os.path.join(PROCESSED_DATA_PATH, "train_texts.pkl"))
        validation_texts.to_pickle(os.path.join(PROCESSED_DATA_PATH, "validation_texts.pkl"))
        test_texts.to_pickle(os.path.join(PROCESSED_DATA_PATH, "test_texts.pkl"))

        torch.save(train_tokens.clone(), os.path.join(PROCESSED_DATA_PATH, "train_tokens.pt"))
        torch.save(validation_tokens.clone(), os.path.join(PROCESSED_DATA_PATH, "validation_tokens.pt"))
        torch.save(test_tokens.clone(), os.path.join(PROCESSED_DATA_PATH, "test_tokens.pt"))

        info = {
            "vocab_size": self.tokenizer.vocab_size,
            "total_ratio": TOTAL_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "max_length_text": MAX_LENGTH_TEXT,
            "max_length": MAX_LENGTH,
        }
        json.dump(info, open(os.path.join(PROCESSED_DATA_PATH, "info.json"), "w"))
        print("Data prepared.")

    def setup(self, stage: str) -> None:
        print("Loading data from disk...")

        # Load data from disk
        self._train_texts = pd.read_pickle(os.path.join(PROCESSED_DATA_PATH, "train_texts.pkl"))["text"].tolist()
        self._validation_texts = pd.read_pickle(os.path.join(PROCESSED_DATA_PATH, "validation_texts.pkl"))[
            "text"
        ].tolist()
        self._test_texts = pd.read_pickle(os.path.join(PROCESSED_DATA_PATH, "test_texts.pkl"))["text"].tolist()

        self._train_tokens = torch.load(os.path.join(PROCESSED_DATA_PATH, "train_tokens.pt")).long()
        self._validation_tokens = torch.load(os.path.join(PROCESSED_DATA_PATH, "validation_tokens.pt")).long()
        self._test_tokens = torch.load(os.path.join(PROCESSED_DATA_PATH, "test_tokens.pt")).long()

        self._train_set = TensorDataset(self._train_tokens)
        self._validation_set = TensorDataset(self._validation_tokens)
        self._test_set = TensorDataset(self._test_tokens)
        print("Data loaded.")

    def train_dataloader(self) -> DataLoader[List[Tensor]]:
        if self._train_texts is None or self._train_tokens is None:
            raise Exception("Please call setup() before using this dataloader.")
        pin_memory = self._device != torch.device("cpu")
        if pin_memory:
            train_loader: DataLoader[List[Tensor]] = DataLoader(  # type: ignore
                self._train_set,
                batch_size=DATA_LOADER_BATCH_SIZE,
                shuffle=True,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device=str(self._device),
            )
        else:
            train_loader = DataLoader(  # type: ignore
                self._train_set,
                batch_size=DATA_LOADER_BATCH_SIZE,
                shuffle=True,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
            )

        return train_loader

    def val_dataloader(self) -> DataLoader[List[Tensor]]:
        if self._validation_set is None or self._validation_tokens is None:
            raise Exception("Please call setup() before using this dataloader.")
        pin_memory = self._device != torch.device("cpu")
        if pin_memory:
            validation_loader: DataLoader[List[Tensor]] = DataLoader(  # type: ignore
                self._validation_set,
                batch_size=DATA_LOADER_BATCH_SIZE,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device=str(self._device),
            )
        else:
            validation_loader = DataLoader(  # type: ignore
                self._validation_set,
                batch_size=DATA_LOADER_BATCH_SIZE,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
            )
        return validation_loader

    def test_dataloader(self) -> DataLoader[List[Tensor]]:
        if self._test_set is None or self._test_tokens is None:
            raise Exception("Please call setup() before using this dataloader.")
        pin_memory = self._device != torch.device("cpu")
        if pin_memory:
            test_loader: DataLoader[List[Tensor]] = DataLoader(  # type: ignore
                self._test_set,
                batch_size=DATA_LOADER_BATCH_SIZE,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device=str(self._device),
            )
        else:
            test_loader = DataLoader(  # type: ignore
                self._test_set,
                batch_size=DATA_LOADER_BATCH_SIZE,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
            )
        return test_loader
