import json
import os
import time

import torch

from core.config import LOCKED
from model.evobrain_load import load_evobrain
from model.train_step import make_optimizer, one_train_step


def make_batch(batch_size, device):
    steps = LOCKED["max_seq_len"]
    nodes = LOCKED["num_nodes"]
    bins = LOCKED["input_dim"]
    input_seq = torch.randn(batch_size, steps, nodes, bins, device=device)
    adj = torch.rand(batch_size, steps, nodes, nodes, device=device)
    adj = adj + adj.transpose(-1, -2)
    seq_lengths = torch.full((batch_size,), steps, device=device, dtype=torch.long)
    labels = torch.zeros(batch_size, device=device)
    labels[: batch_size // 2] = 1.0
    return input_seq, seq_lengths, adj, labels


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise ValueError("need a gpu for this, we got cpu")

    import mamba_ssm

    device = "cuda"
    model = load_evobrain(device=device)
    model = model.to(device)
    optimizer = make_optimizer(model)
    batch_size = LOCKED["train_batch_size"]
    eval_batch = LOCKED["eval_batch_size"]
    pos_weight = 1.0

    n_params = 0
    for p in model.parameters():
        n_params += p.numel()

    # warmup so the kernels are compiled before we time
    for _ in range(10):
        input_seq, seq_lengths, adj, labels = make_batch(batch_size, device)
        one_train_step(model, optimizer, input_seq, seq_lengths, adj, labels, pos_weight)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    n_windows = 0
    start = time.perf_counter()
    for _ in range(50):
        input_seq, seq_lengths, adj, labels = make_batch(batch_size, device)
        one_train_step(model, optimizer, input_seq, seq_lengths, adj, labels, pos_weight)
        n_windows += batch_size
    torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated()
    if peak_bytes < 1:
        raise ValueError("peak mem was 0, timing didnt run")
    card_bytes = torch.cuda.get_device_properties(0).total_memory

    model.eval()
    with torch.no_grad():
        for _ in range(3):
            input_seq, seq_lengths, adj, labels = make_batch(eval_batch, device)
            model(input_seq, seq_lengths, adj)
        torch.cuda.synchronize()
        n_scored = 0
        start = time.perf_counter()
        for _ in range(8):
            input_seq, seq_lengths, adj, labels = make_batch(eval_batch, device)
            model(input_seq, seq_lengths, adj)
            n_scored += eval_batch
        torch.cuda.synchronize()
        score_elapsed = time.perf_counter() - start

    batch128_fits = False
    try:
        input_seq, seq_lengths, adj, labels = make_batch(128, device)
        one_train_step(model, optimizer, input_seq, seq_lengths, adj, labels, pos_weight)
        torch.cuda.synchronize()
        batch128_fits = True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    metrics = {
        "seconds_per_training_sample": train_elapsed / n_windows,
        "train_timed_windows": n_windows,
        "train_elapsed_seconds": train_elapsed,
        "seconds_per_scored_second": score_elapsed / n_scored,
        "eval_batch_size": eval_batch,
        "score_timed_seconds": n_scored,
        "score_elapsed_seconds": score_elapsed,
        "peak_memory_allocated_bytes": peak_bytes,
        "peak_memory_allocated_gib": peak_bytes / (1024 ** 3),
        "concurrent_train_processes_arithmetic": int(card_bytes // peak_bytes),
        "concurrent_train_processes_live": "UNVERIFIED",
        "batch_128_one_step_fits": batch128_fits,
        "parameter_count": n_params,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": card_bytes,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "mamba_ssm_version": mamba_ssm.__version__,
        "train_batch_size": batch_size,
        "command": "python time_cuda.py",
        "end_to_end_wall_clock": "not run, no patient EDFs on this pod yet",
    }

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "p0-gate-metrics.json")
    with open(out, "w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))
