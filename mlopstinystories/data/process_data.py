from datetime import datetime

import torch
from data_module import TinyStories


def main():
    start_time = datetime.now()
    data = TinyStories("data", torch.device("cpu"))
    data.prepare_data()
    end_time = datetime.now()
    print(f"Time to prepare data: {end_time - start_time}")


if __name__ == "__main__":
    main()
