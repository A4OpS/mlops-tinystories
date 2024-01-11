import os
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import TinyStories, TinyStoriesConfig
from device import get_device
from models import TinyStoriesModel, TinyStoriesModelConfig


@dataclass
class TrainModelConfig:
    data_config: TinyStoriesConfig
    model_config: TinyStoriesModelConfig
    max_epochs: int
    max_steps: int
    val_check_interval: int
    limit_val_batches: int
    log_every_n_steps: int


cs = ConfigStore.instance()

cs.store(name="config", node=TrainModelConfig)


@hydra.main(config_path="../conf", config_name="config")
def main(config: TrainModelConfig) -> None:
    device = get_device()

    print(config)

    repo_root = hydra.utils.get_original_cwd()

    data = TinyStories(os.path.join(repo_root, "data"), device.torch(), config.data_config)

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
    )

    trainer.fit(model, datamodule=data)

    model.save("model1")


if __name__ == "__main__":
    main()
