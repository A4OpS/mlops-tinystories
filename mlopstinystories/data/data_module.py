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
    def __init__(self, data_dir: str, device: torch.device) -> None:
        super().__init__()

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # type: ignore
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        if not os.path.exists(self.raw_dir):
            raise Exception(
                "Raw data directory not found. Please place the raw data in '[data_dir]/raw'. \
                    You can use the 'fetch_raw_data.py' script to download the data."
            )
        self.processed_dir = os.path.join(data_dir, "processed")
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        self.device = device

        self.train_texts: Optional[List[str]] = None
        self.validation_texts: Optional[List[str]] = None
        self.test_texts: Optional[List[str]] = None
        self.train_tokens: Optional[Tensor] = None
        self.validation_tokens: Optional[Tensor] = None
        self.test_tokens: Optional[Tensor] = None

    def prepare_data(self) -> None:
        print("Preparing data...")
        # Load data from disk.
        raw_data_path = os.path.join(self.raw_dir, "data.hf")
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
        os.makedirs(self.processed_dir, exist_ok=True)
        train_texts.to_pickle(os.path.join(self.processed_dir, "train_texts.pkl"))
        validation_texts.to_pickle(os.path.join(self.processed_dir, "validation_texts.pkl"))
        test_texts.to_pickle(os.path.join(self.processed_dir, "test_texts.pkl"))

        torch.save(train_tokens.clone(), os.path.join(self.processed_dir, "train_tokens.pt"))
        torch.save(validation_tokens.clone(), os.path.join(self.processed_dir, "validation_tokens.pt"))
        torch.save(test_tokens.clone(), os.path.join(self.processed_dir, "test_tokens.pt"))
        print("Data prepared.")

    def setup(self, stage: str) -> None:
        # Load data from disk
        self.train_texts = pd.read_pickle(os.path.join(self.processed_dir, "train_texts.pkl"))["text"].tolist()
        self.validation_texts = pd.read_pickle(os.path.join(self.processed_dir, "validation_texts.pkl"))[
            "text"
        ].tolist()
        self.test_texts = pd.read_pickle(os.path.join(self.processed_dir, "test_texts.pkl"))["text"].tolist()

        self.train_tokens = torch.load(os.path.join(self.processed_dir, "train_tokens.pt"))
        self.validation_tokens = torch.load(os.path.join(self.processed_dir, "validation_tokens.pt"))
        self.test_tokens = torch.load(os.path.join(self.processed_dir, "test_tokens.pt"))

        self.train_set = TensorDataset(self.train_tokens)
        self.validation_set = TensorDataset(self.validation_tokens)
        self.test_set = TensorDataset(self.test_tokens)

    def train_dataloader(self) -> DataLoader[List[Tensor]]:
        if self.train_texts is None or self.train_tokens is None:
            raise Exception("Please call setup() before using this dataloader.")
        pin_memory = self.device != torch.device("cpu")
        if pin_memory:
            train_loader: DataLoader[List[Tensor]] = DataLoader(  # type: ignore
                self.train_set,
                batch_size=64,
                shuffle=True,
                num_workers=torch.get_num_threads(),
                pin_memory=True,
                pin_memory_device=str(self.device),
            )
        else:
            train_loader = DataLoader(  # type: ignore
                self.train_set,
                batch_size=64,
                shuffle=True,
                num_workers=torch.get_num_threads(),
            )

        return train_loader

    def val_dataloader(self) -> DataLoader[List[Tensor]]:
        if self.validation_set is None or self.validation_tokens is None:
            raise Exception("Please call setup() before using this dataloader.")
        pin_memory = self.device != torch.device("cpu")
        if pin_memory:
            validation_loader: DataLoader[List[Tensor]] = DataLoader(  # type: ignore
                self.validation_set,
                batch_size=64,
                num_workers=torch.get_num_threads(),
                pin_memory=True,
                pin_memory_device=str(self.device),
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
        pin_memory = self.device != torch.device("cpu")
        if pin_memory:
            test_loader: DataLoader[List[Tensor]] = DataLoader(  # type: ignore
                self.test_set,
                batch_size=64,
                num_workers=torch.get_num_threads(),
                pin_memory=True,
                pin_memory_device=str(self.device),
            )
        else:
            test_loader = DataLoader(  # type: ignore
                self.test_set,
                batch_size=64,
                num_workers=torch.get_num_threads(),
            )
        return test_loader
