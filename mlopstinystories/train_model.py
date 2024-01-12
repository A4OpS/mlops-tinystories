
import hydra
from hydra.core.config_store import ConfigStore

from mlopstinystories.train import TrainModelConfig, train_model

cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainModelConfig)

@hydra.main(config_path="../conf/train", version_base="1.3")
def main(config: TrainModelConfig) -> None:
    repo_root = hydra.utils.get_original_cwd()
    train_model(repo_root, config)


if __name__ == "__main__":
    main()
