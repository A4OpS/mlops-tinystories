import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from train_model import TrainModelConfig, train_model

cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainModelConfig)


@hydra.main(config_path="../conf/train", version_base="1.3")
def main(config: TrainModelConfig) -> None:
    profiler = PyTorchProfiler(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        row_limit=20,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        sort_by_key="cpu_time_total",
        on_trace_ready=tensorboard_trace_handler("./profilers"),
    )

    train_model(config, profiler)


if __name__ == "__main__":
    main()
