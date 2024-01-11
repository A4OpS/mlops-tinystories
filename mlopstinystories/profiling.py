import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from train_model import main

with profile(
        activities=[ProfilerActivity.CPU], 
        record_shapes=True, 
        profile_memory = True,
        on_trace_ready=tensorboard_trace_handler("./profilers")) as prof:
    main()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

#prof.export_chrome_trace("trace.json")