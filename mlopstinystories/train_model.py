from device import get_device
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data import TinyStories
from models import TinyStoriesConfig, TinyStoriesModel

if __name__ == "__main__":
    device = get_device()

    data = TinyStories("data", device.torch())

    config = TinyStoriesConfig(
        num_layers=2,
        intermediate_size=512,
        hidden_size=512,
        num_heads=8,
        vocab_size=data.vocab_size,
        max_position_embeddings=512,
    )

    model = TinyStoriesModel.initialize(config, device.torch())
    print("Device: ", model.device())
    print(f"Number of parameters: {model.num_params()}")

    trainer = Trainer(
        logger=WandbLogger(project="mlops-tinystories"), accelerator=device.lightning(), max_epochs=1, max_steps=10
    )

    trainer.fit(model, datamodule=data)

    model.save("model1")
