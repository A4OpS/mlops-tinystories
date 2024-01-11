from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler

from data import TinyStories
from device import get_device
from models import TinyStoriesConfig, TinyStoriesModel

def main():
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

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        save_top_k=3,
        save_on_train_epoch_end=False,
        monitor="val_loss",
        filename="{step}-{val_loss:.2f}",
        mode="min",
    )

    profiler = PyTorchProfiler(
        activities=[ProfilerActivity.CPU],
        row_limit = 20,
        record_shapes = True,
        with_stack = True,
        profile_memory = True,
        #export_to_chrome_tracing = True,
        sort_by_key = "cpu_time_total",
        on_trace_ready = tensorboard_trace_handler("./profilers")
    )

    trainer = Trainer(
        logger=WandbLogger(project="mlops-tinystories"),
        accelerator=device.lightning(),
        max_epochs=1,
        callbacks=[checkpoint_callback],
        max_steps=50,
        val_check_interval=5,
        limit_val_batches=10,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        #profiler = profiler,
    )

    trainer.fit(model, datamodule=data)

    model.save("model1")


if __name__ == "__main__":
    main()