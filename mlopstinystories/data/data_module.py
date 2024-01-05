import os
from typing import List, Optional, Tuple

import datasets
import pandas as pd
import torch
from datasets import DatasetDict
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

VALIDATION_RATIO = 0.1
TEST_RATIO = 0.05


class TinyStories(LightningDataModule):
    def __init__(self, data_dir: str, device: torch.device) -> None:
        super().__init__()
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

        self.train_set: Optional[TensorDataset] = None
        self.test_set: Optional[TensorDataset] = None

    def prepare_data(self) -> None:
        # Load data from disk.
        raw_data_path = os.path.join(self.raw_dir, "data.hf")
        if not os.path.exists(raw_data_path):
            raise Exception("Raw data not found. Please place it in '[data_dir]/raw'.")
        raw_data: DatasetDict = datasets.load_from_disk(raw_data_path)  # type: ignore
        # Convert dataset splits to pandas dataframes.
        train_data = raw_data["train"].data.to_pandas()
        validation_data = raw_data["validation"].data.to_pandas()
        # Concatenate the dataframes.
        data = pd.concat([train_data, validation_data])

        shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        test_end_index = int(len(shuffled_data) * TEST_RATIO)
        validation_end_index = int(len(shuffled_data) * (TEST_RATIO + VALIDATION_RATIO))

        test_data = shuffled_data[0:test_end_index]
        validation_data = shuffled_data[test_end_index:validation_end_index]
        train_data = shuffled_data[validation_end_index:]
        # Split the data into train, validation and test sets.

        # Save the data to disk.
        os.makedirs(self.processed_dir, exist_ok=True)
        train_data.to_pickle(os.path.join(self.processed_dir, "train_data.pkl"))
        validation_data.to_pickle(os.path.join(self.processed_dir, "validation_data.pkl"))
        test_data.to_pickle(os.path.join(self.processed_dir, "test_data.pkl"))

    def setup(self, stage: str) -> None:
        # Load data from disk
        self.train_data = pd.read_pickle(os.path.join(self.processed_dir, "train_data.pkl"))
        self.validation_data = pd.read_pickle(os.path.join(self.processed_dir, "validation_data.pkl"))
        self.test_data = pd.read_pickle(os.path.join(self.processed_dir, "test_data.pkl"))

        return

    def train_dataloader(self) -> DataLoader[List[Tensor]]:
        if self.train_set is None:
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

    def test_dataloader(self) -> DataLoader[List[Tensor]]:
        if self.test_set is None:
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
