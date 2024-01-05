import logging
import os
import warnings

import datasets
import pandas as pd
from datasets.dataset_dict import DatasetDict

if __name__ == "__main__":
    log = logging.getLogger(__name__)

    # Get dataset from huggingface
    log.info("Getting TinyStories dataset from HuggingFace...")
    warnings.filterwarnings("ignore", message="Repo card metadata block was not found. Setting CardData to empty.")
    dataset: DatasetDict = datasets.load_dataset("roneneldan/TinyStories")  # type: ignore
    # Convert dataset splits to pandas dataframes.
    log.info("Converting dataset to pandas dataframes...")
    train_data = dataset["train"].data.to_pandas()
    validation_data = dataset["validation"].data.to_pandas()
    # Add columns to indicate whether the data was originally meant for validation.
    # We will likely not care about this, but I don't like losing information.
    train_data["validation"] = False
    validation_data["validation"] = True
    # Concatenate the dataframes.
    data = pd.concat([train_data, validation_data])

    # Save the data to disk.
    log.info("Saving data to disk...")
    data_dir = "data"
    raw_data_dir = os.path.join("data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    data.to_pickle(os.path.join(raw_data_dir, "data.pkl"))
    log.info("Done.")
