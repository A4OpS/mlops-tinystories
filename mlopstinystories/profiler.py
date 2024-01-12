from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler

profiler = PyTorchProfiler(
        activities=[ProfilerActivity.CPU],
        row_limit = 20,
        record_shapes = True,
        with_stack = True,
        profile_memory = True,
        sort_by_key = "cpu_time_total",
        on_trace_ready = tensorboard_trace_handler("./profilers")
    )