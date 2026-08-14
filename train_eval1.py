import json
import os
import time

import torch

from config import LOCKED
from evobrain_load import load_evobrain
from run import train_packed_eval_run
from time_cuda import make_batch
from train_step import make_optimizer, one_train_step


PACKED_FOLDERS = (
    (
        "experiments/runs/within_patient/sub-004/pearson/sop30/seed123/eval1",
        True,
    ),
    (
        "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
        False,
    ),
    (
        "experiments/runs/within_patient/sub-113/pearson/sop30/seed123/eval1",
        False,
    ),
)


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

    here = os.path.dirname(os.path.abspath(__file__))
    summary = {
        "command": "python train_eval1.py",
        "compile_seconds": float(compile_seconds),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "mamba_ssm_version": mamba_ssm.__version__,
        "patients": [],
    }
    wall_start = time.perf_counter()
    for rel, replace_existing_train in PACKED_FOLDERS:
        folder = os.path.join(here, rel)
        model = load_evobrain(device=device)
        model = model.to(device)
        optimizer = make_optimizer(model)
        extra = {
            "train_model": "evobrain",
            "train_torch_version": torch.__version__,
            "command": "python train_eval1.py",
            "seed": LOCKED["seed"],
            "dataset_accession": LOCKED["dataset_accession"],
            "dataset_version": LOCKED["dataset_version"],
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "cuda_version": torch.version.cuda,
            "mamba_ssm_version": mamba_ssm.__version__,
            "train_batch_size": LOCKED["train_batch_size"],
            "eval_batch_size": LOCKED["eval_batch_size"],
        }
        _folder, metrics = train_packed_eval_run(
            folder,
            model,
            optimizer,
            extra=extra,
            device=device,
            compile_seconds=compile_seconds,
            replace_existing_train=replace_existing_train,
        )
        row = {
            "patient_id": metrics.get("patient_id"),
            "best_epoch": metrics.get("best_epoch"),
            "best_monitor_bce": metrics.get("best_monitor_bce"),
            "train_seconds": metrics.get("train_seconds"),
            "score_seconds": metrics.get("score_seconds"),
            "n_seconds_scored": metrics.get("n_seconds_scored"),
            "n_test_windows_scored": metrics.get("n_test_windows_scored"),
            "n_validation_windows_scored": metrics.get("n_validation_windows_scored"),
            "pos_weight": metrics.get("pos_weight"),
            "used_stored_ends": metrics.get("used_stored_ends"),
            "scored_at_train_stride": metrics.get("scored_at_train_stride"),
            "parameter_count": metrics.get("parameter_count"),
        }
        summary["patients"].append(row)
        print("folder", folder)
        print(json.dumps(row, indent=2))
    summary["wall_seconds"] = float(time.perf_counter() - wall_start)
    out = os.path.join(here, "p2-train-eval1-summary.json")
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
