import os
import sys
from glob import glob
from typing import List

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast


from data import RAW_DATA_PATH, PROCESSED_DATA_PATH, TinyStories, TinyStoriesConfig, fetch_raw_data



# Test fetching of raw data
def fetch_raw_data_test():
    fetch_raw_data()
    # Test files in data directory
    assert os.path.exists(RAW_DATA_PATH), "Raw data directory does not exists"
    assert os.path.exists(
        os.path.join(RAW_DATA_PATH, "train")
    ), f"Missing raw train directory '{os.path.join(RAW_DATA_PATH, 'train')}'"
    assert os.path.exists(os.path.join(RAW_DATA_PATH, "validation")), "Missing raw validation directory"
    assert os.path.exists(os.path.join(RAW_DATA_PATH, "dataset_dict.json")), "Missing raw dataset dict file"

    # Test files in train directory
    assert len(glob(os.path.join(RAW_DATA_PATH, "train", "*.arrow"))) == 4, "Missing raw train files"
    assert os.path.exists(os.path.join(RAW_DATA_PATH, "train", "dataset_info.json")), "Missing raw dataset info file"
    assert os.path.exists(os.path.join(RAW_DATA_PATH, "train", "state.json")), "Missing raw dataset info file"

    # Test if data train file are not empty
    files_train = os.listdir(os.path.join(RAW_DATA_PATH, "train"))
    for file in files_train:
        assert os.path.getsize(os.path.join(RAW_DATA_PATH, "train", file)) > 0, f"Processed data file {file} is empty"

    # Test files in validation directory
    assert len(glob(os.path.join(RAW_DATA_PATH, "validation", "*.arrow"))) == 1, "Missing raw validation files"
    assert os.path.exists(
        os.path.join(RAW_DATA_PATH, "validation", "dataset_info.json")
    ), "Missing raw dataset info file"
    assert os.path.exists(os.path.join(RAW_DATA_PATH, "validation", "state.json")), "Missing raw dataset info file"

    # Test if datafile are not empty
    files_validation = os.listdir(os.path.join(RAW_DATA_PATH, "validation"))
    for file in files_validation:
        assert (
            os.path.getsize(os.path.join(RAW_DATA_PATH, "validation", file)) > 0
        ), f"Processed data file {file} is empty"

    # Possible test to do: [https://github.com/albertsgarde/mlops-tinystories/issues/42]
    #     - Test if the dataset_dict.json corresponds to the subfolders in data/raw
    #     - Test if the state.json has the correct number of files in both train and validation folder



# Test loading/fetching of data module
def data_module_test():
    def dataloader_test(data_loader: DataLoader[List[torch.Tensor]]):
        assert isinstance(data_loader, DataLoader), "Data loader is not of type torch.utils.data.DataLoader"
        assert len(data_loader) > 0, "Data loader is empty"

    config = TinyStoriesConfig(
        total_ratio=0.005,
        validation_ratio=0.1,
        test_ratio=0.05,
        max_length_text=1024,
        max_length=256,
        data_loader_batch_size=4,
    )
    data = TinyStories("", torch.device("cpu"), config)
    data.prepare_data()
    data.setup("fisk")

    # Test if processed data directory exists
    assert os.path.exists(PROCESSED_DATA_PATH), "Processed data directory does not exists"

    # Test if processed data files exists
    assert os.path.exists(os.path.join(PROCESSED_DATA_PATH, "test_texts.pkl")), "Missing processed test texts file"
    assert os.path.exists(os.path.join(PROCESSED_DATA_PATH, "train_texts.pkl")), "Missing processed train texts file"
    assert os.path.exists(
        os.path.join(PROCESSED_DATA_PATH, "validation_texts.pkl")
    ), "Missing processed validation texts file"

    assert os.path.exists(os.path.join(PROCESSED_DATA_PATH, "test_tokens.pt")), "Missing processed test token file"
    assert os.path.exists(os.path.join(PROCESSED_DATA_PATH, "train_tokens.pt")), "Missing processed train token file"
    assert os.path.exists(
        os.path.join(PROCESSED_DATA_PATH, "validation_tokens.pt")
    ), "Missing processed validation token file"

    # Test if processed data files are not empty
    files = os.listdir(PROCESSED_DATA_PATH)
    for file in files:
        assert os.path.getsize(os.path.join(PROCESSED_DATA_PATH, file)) > 0, f"Processed data file {file} is empty"

    # Test data module
    assert isinstance(data, LightningDataModule), "Data module is not of type LightningDataModule"

    # Test tokenizer
    assert isinstance(data.tokenizer, PreTrainedTokenizerFast), "Tokenizer is not of type PreTrainedTokenizerFast"
    assert data._vocab_size > 0, "Tokenizer vocab is empty"

    # Test dataloaders
    dataloader_test(data.train_dataloader())
    dataloader_test(data.val_dataloader())
    dataloader_test(data.test_dataloader())


def test_data():
    fetch_raw_data_test()
    data_module_test()

