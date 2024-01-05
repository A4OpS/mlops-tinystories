import logging
import os
import warnings

import datasets
from datasets.dataset_dict import DatasetDict

if __name__ == "__main__":
    log = logging.getLogger(__name__)

    # Get dataset from huggingface
    log.info("Getting TinyStories dataset from HuggingFace...")
    warnings.filterwarnings("ignore", message="Repo card metadata block was not found. Setting CardData to empty.")
    dataset: DatasetDict = datasets.load_dataset("roneneldan/TinyStories")  # type: ignore
    dataset.save_to_disk(os.path.join("data", "raw", "data.hf"))
