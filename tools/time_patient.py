import json
import os
import time

import torch

from core.config import LOCKED
from evaluation.metrics import load_gate_metrics
from model.evobrain_load import load_evobrain
from run import timed_patient_run
from tools.time_cuda import make_batch
from model.train_step import make_optimizer, one_train_step


def main():
    if not torch.cuda.is_available():
        raise ValueError("need a gpu for this, we got cpu")

    import mamba_ssm

    device = "cuda"
    compile_start = time.perf_counter()
    model = load_evobrain(device=device)
    model = model.to(device)
    optimizer = make_optimizer(model)
    input_seq, seq_lengths, adj, labels = make_batch(LOCKED["train_batch_size"], device)
    one_train_step(model, optimizer, input_seq, seq_lengths, adj, labels, 1.0)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start
    # compile dirty the weights, so load a fresh model for the timed run
    model = load_evobrain(device=device)
    model = model.to(device)
    optimizer = make_optimizer(model)

    extra = {
        "train_model": "evobrain",
        "train_torch_version": torch.__version__,
        "command": "python time_patient.py",
        "seed": LOCKED["seed"],
        "dataset_accession": LOCKED["dataset_accession"],
        "dataset_version": LOCKED["dataset_version"],
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_version": torch.version.cuda,
        "mamba_ssm_version": mamba_ssm.__version__,
        "train_batch_size": LOCKED["train_batch_size"],
        "eval_batch_size": LOCKED["eval_batch_size"],
    }
    gate_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "p0-gate-metrics.json")
    gate = load_gate_metrics(gate_path)
    for key in gate:
        extra[key] = gate[key]

    base = "data/seizeit2/sub-004/ses-01"
    folder, metrics = timed_patient_run(
        ".",
        "sub-004",
        f"{base}/eeg/sub-004_ses-01_task-szMonitoring_run-01_eeg.edf",
        f"{base}/ecg/sub-004_ses-01_task-szMonitoring_run-01_ecg.edf",
        f"{base}/emg/sub-004_ses-01_task-szMonitoring_run-01_emg.edf",
        f"{base}/mov/sub-004_ses-01_task-szMonitoring_run-01_mov.edf",
        f"{base}/eeg/sub-004_ses-01_task-szMonitoring_run-01_events.tsv",
        model,
        optimizer,
        extra=extra,
        device=device,
        compile_seconds=compile_seconds,
    )
    print("folder", folder)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
