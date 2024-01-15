from dataclasses import dataclass
from datetime import datetime

import hydra
import torch
from data_module import TinyStories, TinyStoriesConfig
from hydra.core.config_store import ConfigStore


@dataclass
class ProcessDataConfig:
    data_config: TinyStoriesConfig


cs = ConfigStore.instance()

cs.store(name="process_data_config", node=ProcessDataConfig)


@hydra.main(config_path="../../conf/process_data", version_base="1.3")
def main(config: ProcessDataConfig) -> None:
    start_time = datetime.now()
    repo_root = hydra.utils.get_original_cwd()
    data = TinyStories(repo_root, torch.device("cpu"), config.data_config)
    data.prepare_data()
    end_time = datetime.now()
    print(f"Time to prepare data: {end_time - start_time}")


if __name__ == "__main__":
    main()
