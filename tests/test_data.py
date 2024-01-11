import os
from glob import glob

import torch
from pytorch_lightning import LightningDataModule
from transformers import PreTrainedTokenizerFast

from mlopstinystories.data import PROCESSED_DATA_PATH, RAW_DATA_PATH, TinyStories


# Test fetching of raw data
def test_fetch_raw_data():
    # Test files in data directory
    assert os.path.exists(RAW_DATA_PATH), "Raw data directory does not exists"
    assert os.path.exists(os.path.join(RAW_DATA_PATH, "train")), "Missing raw train directory"
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


# Test processing of raw data
def test_process_data():
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


#def data_loader_test(DataLoader[List[Tensor]]):
#    pass

# Test loading/fetching of data module
def test_data_module():
    data = TinyStories(torch.device("cpu"))
    assert isinstance(data, LightningDataModule), "Data module is not of type LightningDataModule"

    # Test tokenizer
    assert isinstance(data.tokenizer, PreTrainedTokenizerFast), "Tokenizer is not of type PreTrainedTokenizerFast"
    assert data._vocab_size > 0, "Tokenizer vocab is empty"

    # Test setup

    # Test dataloaders
    #data_loader_test(data.train_dataloader)
    #data_loader_test(data.val_dataloader)
    #data_loader_test(data.test_dataloader)
