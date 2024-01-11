import logging
import os
import warnings

import datasets
from datasets.dataset_dict import DatasetDict

RAW_DATA_PATH: str = os.path.join("data", "raw")


def fetch_raw_data():
    log = logging.getLogger(__name__)

    # Get dataset from huggingface
    log.info("Getting TinyStories dataset from HuggingFace...")
    warnings.filterwarnings("ignore", message="Repo card metadata block was not found. Setting CardData to empty.")
    dataset: DatasetDict = datasets.load_dataset("roneneldan/TinyStories")  # type: ignore
    dataset.save_to_disk(RAW_DATA_PATH)


if __name__ == "__main__":
    fetch_raw_data()
