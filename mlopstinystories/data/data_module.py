import os
from dataclasses import dataclass
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


@dataclass
class TinyStoriesConfig:
    """
    Configuration for the TinyStories dataset.

    Attributes:
        total_ratio (float): The ratio of the total dataset to use.
            Setting this to 1 will use the entire dataset, resulting in 4 GB of processed data
            and will likely eat _all_ your RAM, so be careful with setting this too high.
        validation_ratio (float): The ratio of use data to use for validation.
            The validation set will be `total_ratio * validation_ratio` of the total dataset.
        test_ratio (float): The ratio of use data to use for testing.
            The test set will be `total_ratio * test_ratio` of the total dataset.
        max_length_text (int): The maximum length of the text in the dataset.
            Texts longer than this will be discarded. If there are few than `total_ratio` texts shorter than this,
            `total_ratio` will be ignored.
        max_length (int): The maximum length of the tokenized text. Texts longer than this will be truncated.
            It can therefore be a good idea to set this low enough that few texts shorter than `max_length_text`
            have more tokens than `max_length`.
        data_loader_batch_size (int): The batch size to use for the dataloaders.
    """

    total_ratio: float
    validation_ratio: float
    test_ratio: float
    max_length_text: int
    max_length: int
    data_loader_batch_size: int


class TinyStories(LightningDataModule):
    _config: TinyStoriesConfig

    _raw_data_path: str
    _processed_data_path: str

    _tokenizer: PreTrainedTokenizerFast
    _vocab_size: int

    _device: torch.device

    _train_texts: Optional[List[str]]
    _validation_texts: Optional[List[str]]
    _test_texts: Optional[List[str]]
    _train_tokens: Optional[Tensor]
    _validation_tokens: Optional[Tensor]
    _test_tokens: Optional[Tensor]

    def __init__(self, root_path: str, device: torch.device, config: TinyStoriesConfig) -> None:
        super().__init__()

        self._config = config

        self._raw_data_path = os.path.join(root_path, RAW_DATA_PATH)
        self._processed_data_path = os.path.join(root_path, PROCESSED_DATA_PATH)

        self._tokenizer = TinyStories.create_tokenizer()
        self._vocab_size = self.tokenizer.vocab_size

        if not os.path.exists(self._raw_data_path):
            raise Exception(
                f"Raw data directory not found. Please place the raw data in [{self._raw_data_path}] "
                f"from 'data/constants.py'. You can use the 'fetch_raw_data.py' script to download the data. "
            )

        if not os.path.exists(self._processed_data_path):
            os.makedirs(self._processed_data_path)
        self._device = device

        self._train_texts: Optional[List[str]] = None
        self._validation_texts: Optional[List[str]] = None
        self._test_texts: Optional[List[str]] = None
        self._train_tokens: Optional[Tensor] = None
        self._validation_tokens: Optional[Tensor] = None
        self._test_tokens: Optional[Tensor] = None

    @staticmethod
    def create_tokenizer() -> PreTrainedTokenizerFast:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @property
    def config(self) -> TinyStoriesConfig:
        return self._config

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def prepare_data(self) -> None:
        print("Preparing data...")
        if not os.path.exists(self._raw_data_path):
            raise Exception(
                f"Raw data directory not found. Please place the raw data in [{self._raw_data_path}] \
                    from 'data/constants.py'. You can use the 'fetch_raw_data.py' script to download the data."
            )
        # Load data from disk.
        raw_data: DatasetDict = datasets.load_from_disk(self._raw_data_path)  # type: ignore
        # Convert dataset splits to pandas dataframes.
        train_texts = raw_data["train"].data.to_pandas()
        validation_texts = raw_data["validation"].data.to_pandas()
        # Concatenate the dataframes.
        data = pd.concat([train_texts, validation_texts])

        # Shuffle the data and reset the index.
        short = data[data["text"].apply(len) <= self.config.max_length_text]
        short_ratio = len(short) / len(data)
        sample_ratio = min(1, self.config.total_ratio / short_ratio)
        shuffled_data = short.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)

        strings = shuffled_data["text"].tolist()

        tokenized_strings = self.tokenizer(
            strings,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.config.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens: Tensor = tokenized_strings["input_ids"].to(torch.int32)  # type: ignore

        # Split the data into train, validation and test sets.
        test_end_index = int(len(shuffled_data) * self.config.test_ratio)
        validation_end_index = int(len(shuffled_data) * (self.config.test_ratio + self.config.validation_ratio))

        test_texts = shuffled_data[0:test_end_index]
        validation_texts = shuffled_data[test_end_index:validation_end_index]
        train_texts = shuffled_data[validation_end_index:]

        test_tokens = tokens[0:test_end_index]
        validation_tokens = tokens[test_end_index:validation_end_index]
        train_tokens = tokens[validation_end_index:]

        # Save the data to disk.
        os.makedirs(self._processed_data_path, exist_ok=True)
        train_texts.to_pickle(os.path.join(self._processed_data_path, "train_texts.pkl"))
        validation_texts.to_pickle(os.path.join(self._processed_data_path, "validation_texts.pkl"))
        test_texts.to_pickle(os.path.join(self._processed_data_path, "test_texts.pkl"))

        torch.save(train_tokens.clone(), os.path.join(self._processed_data_path, "train_tokens.pt"))
        torch.save(validation_tokens.clone(), os.path.join(self._processed_data_path, "validation_tokens.pt"))
        torch.save(test_tokens.clone(), os.path.join(self._processed_data_path, "test_tokens.pt"))

        print("Data prepared.")

    def setup(self, stage: str) -> None:
        print("Loading data from disk...")

        # Load data from disk
        self._train_texts = pd.read_pickle(os.path.join(self._processed_data_path, "train_texts.pkl"))["text"].tolist()
        self._validation_texts = pd.read_pickle(os.path.join(self._processed_data_path, "validation_texts.pkl"))[
            "text"
        ].tolist()
        self._test_texts = pd.read_pickle(os.path.join(self._processed_data_path, "test_texts.pkl"))["text"].tolist()

        self._train_tokens = torch.load(os.path.join(self._processed_data_path, "train_tokens.pt")).long()
        self._validation_tokens = torch.load(os.path.join(self._processed_data_path, "validation_tokens.pt")).long()
        self._test_tokens = torch.load(os.path.join(self._processed_data_path, "test_tokens.pt")).long()

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
                batch_size=self.config.data_loader_batch_size,
                shuffle=True,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device=str(self._device),
            )
        else:
            train_loader = DataLoader(  # type: ignore
                self._train_set,
                batch_size=self.config.data_loader_batch_size,
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
                batch_size=self.config.data_loader_batch_size,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device=str(self._device),
            )
        else:
            validation_loader = DataLoader(  # type: ignore
                self._validation_set,
                batch_size=self.config.data_loader_batch_size,
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
                batch_size=self.config.data_loader_batch_size,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device=str(self._device),
            )
        else:
            test_loader = DataLoader(  # type: ignore
                self._test_set,
                batch_size=self.config.data_loader_batch_size,
                num_workers=torch.get_num_threads(),
                persistent_workers=True,
            )
        return test_loader
