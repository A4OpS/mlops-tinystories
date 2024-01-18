from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

from data import TinyStories, TinyStoriesConfig
from device import get_device
from models import TinyStoriesModel, TinyStoriesModelConfig


@dataclass
class TrainModelConfig:
    data_config: TinyStoriesConfig = TinyStoriesConfig()
    model_config: TinyStoriesModelConfig = TinyStoriesModelConfig()
    max_epochs: int = 1
    max_steps: int = 50
    val_check_interval: int = 5
    limit_val_batches: int = 10
    log_every_n_steps: int = 1


cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainModelConfig)


def train_model(config: TrainModelConfig, profiler:Optional[PyTorchProfiler] = None) -> None:
    device = get_device()

    repo_root = hydra.utils.get_original_cwd()

    data = TinyStories(repo_root, device.torch(), config.data_config)

    model = TinyStoriesModel.initialize(config.model_config, device.torch())
    print("Device: ", model.device())
    print(f"Number of parameters: {model.num_params()}")

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        save_top_k=3,
        save_on_train_epoch_end=False,
        monitor="val_loss",
        filename="{step}-{val_loss:.2f}",
        mode="min",
    )
    trainer = Trainer(
        logger=WandbLogger(project="mlops-tinystories"),
        accelerator=device.lightning(),
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback],
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        limit_val_batches=config.limit_val_batches,
        log_every_n_steps=config.log_every_n_steps,
        num_sanity_val_steps=2,
        profiler = profiler,
    )

    trainer.fit(model, datamodule=data)

    model.save("model1")

@hydra.main(config_path="../conf/train", version_base="1.3")
def main(config: TrainModelConfig) -> None:
    train_model(config)

if __name__ == "__main__":
    main()
