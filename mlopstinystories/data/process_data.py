import logging

import torch
from data_module import TinyStories

if __name__ == "__main__":
    log = logging.getLogger(__name__)

    # Get dataset from huggingface
    data = TinyStories("data", torch.device("cpu"))
    data.prepare_data()
