import json
import os
import time

import torch

from alarms import save_window_scores
from core.config import LOCKED
from epoch_trace import install_logged_train_loop
from evobrain_load import load_evobrain
from metrics import read_metrics, replace_metrics_keys
from packs.pack import load_split
from run import train_packed_eval_run
from time_cuda import make_batch
from train_loop import score_windows
from train_step import make_optimizer, one_train_step


PACKED_FOLDERS = (
    "experiments/runs/within_patient/sub-051/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
)


def main():
    if not torch.cuda.is_available():
        raise ValueError("need a gpu for this, we got cpu")

    import mamba_ssm

    install_logged_train_loop()
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
        "command": "python honest_retrain.py",
        "compile_seconds": float(compile_seconds),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "mamba_ssm_version": mamba_ssm.__version__,
        "patients": [],
    }
    wall_start = time.perf_counter()
    for rel in PACKED_FOLDERS:
        folder = os.path.join(here, rel)
        model = load_evobrain(device=device)
        model = model.to(device)
        optimizer = make_optimizer(model)
        extra = {
            "train_model": "evobrain",
            "train_torch_version": torch.__version__,
            "command": "python honest_retrain.py",
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
            replace_existing_train=True,
        )
        _save_train_scores(folder, model, device)
        epoch_extra = _epoch_keys(folder)
        replace_metrics_keys(folder, epoch_extra)
        metrics = read_metrics(folder)
        row = {
            "patient_id": metrics.get("patient_id"),
            "best_epoch": metrics.get("best_epoch"),
            "n_epochs_run": metrics.get("n_epochs_run"),
            "best_monitor_bce": metrics.get("best_monitor_bce"),
            "logit_zero_monitor_bce": metrics.get("logit_zero_monitor_bce"),
            "train_seconds": metrics.get("train_seconds"),
            "score_seconds": metrics.get("score_seconds"),
            "n_monitor_windows_scored": metrics.get("n_monitor_windows_scored"),
            "n_test_windows_scored": metrics.get("n_test_windows_scored"),
            "n_validation_windows_scored": metrics.get("n_validation_windows_scored"),
            "pos_weight": metrics.get("pos_weight"),
        }
        summary["patients"].append(row)
        print("folder", folder)
        print(json.dumps(row, indent=2))
    summary["wall_seconds"] = float(time.perf_counter() - wall_start)
    out = os.path.join(here, "p2-honest-retrain-gpu.json")
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def _save_train_scores(folder, model, device):
    train_f, train_g, _train_y, train_ends = load_split(os.path.join(folder, "train.npz"))
    if train_ends is None:
        raise ValueError("train pack has no stored ends")
    logits = score_windows(model, train_f, train_g, device=device)
    save_window_scores(
        os.path.join(folder, "train_scores.npz"),
        train_ends,
        logits.numpy(),
    )


def _epoch_keys(folder):
    path = os.path.join(folder, "checkpoints", "epoch_log.json")
    if not os.path.isfile(path):
        raise ValueError(f"epoch log is missing: {path}")
    with open(path) as handle:
        blob = json.load(handle)
    if blob.get("n_epochs_logged", 0) < 1:
        raise ValueError("epoch log has no rows")
    return {
        "logit_zero_monitor_bce": blob["logit_zero_monitor_bce"],
        "epoch_log": blob["epochs"],
        "n_epochs_logged": blob["n_epochs_logged"],
    }


if __name__ == "__main__":
    main()
