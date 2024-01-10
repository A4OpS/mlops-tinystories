import logging
from datetime import datetime

import torch
from data_module import TinyStories, TinyStoriesConfig

if __name__ == "__main__":
    log = logging.getLogger(__name__)

    start_time = datetime.now()
    config = TinyStoriesConfig()
    data = TinyStories("data", torch.device("cpu"), config)
    data.prepare_data()
    end_time = datetime.now()
    print(f"Time to prepare data: {end_time - start_time}")
