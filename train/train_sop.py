import json
import os
import sys
import time
import traceback

import torch

from evaluation.alarms import save_window_scores
from graphs.attention import load_attention_evobrain, prepare_attention_folder
from core.config import ALLOWED_SOP_MINUTES, LOCKED, run_folder_path
from evaluation.epoch_trace import install_logged_train_loop
from model.evobrain_load import load_evobrain
from evaluation.metrics import read_metrics, replace_metrics_keys
from packs.pack import load_split
from run import train_packed_eval_run
from tools.time_cuda import make_batch
from model.train_loop import score_windows
from model.train_step import make_optimizer, one_train_step

PATIENTS = ("sub-004", "sub-051", "sub-052", "sub-113", "sub-039", "sub-073", "sub-015")
GRAPH_STRATEGIES = ("pearson", "mi", "attention")
NEW_SOP_MINUTES = (15, 5)
SUMMARY_NAME = "results/p4-sop-runs-gpu.json"


def parse_args(argv):
    if len(argv) == 1:
        grid = []
        for sop_minutes in NEW_SOP_MINUTES:
            for patient_id in PATIENTS:
                for graph_strategy in GRAPH_STRATEGIES:
                    grid.append((patient_id, graph_strategy, sop_minutes))
        return grid
    if len(argv) == 4:
        try:
            sop_minutes = int(argv[3])
        except ValueError:
            raise ValueError(f"sop_minutes has to be a whole number, we got {argv[3]!r}")
        if sop_minutes not in ALLOWED_SOP_MINUTES:
            raise ValueError(
                "sop_minutes has to be 30, 15, 10, or 5, "
                f"we got {sop_minutes!r}"
            )
        return [(argv[1], argv[2], sop_minutes)]
    raise ValueError(
        "usage: python train_sop.py [patient_id graph_strategy sop_minutes]"
    )


def main():
    if not torch.cuda.is_available():
        raise ValueError("need a gpu for this, we got cpu")
    grid = parse_args(sys.argv)
    # mamba_ssm is only installed on the gpu pod
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

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary = {
        "command": "python train_sop.py",
        "compile_seconds": float(compile_seconds),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "mamba_ssm_version": mamba_ssm.__version__,
        "runs": [],
    }
    wall_start = time.perf_counter()
    for patient_id, graph_strategy, sop_minutes in grid:
        row = _train_one_run(
            here, patient_id, graph_strategy, sop_minutes, device, compile_seconds
        )
        summary["runs"].append(row)
        print(json.dumps(row, indent=2))
        _write_summary(here, summary, wall_start)
    _write_summary(here, summary, wall_start)
    print(json.dumps(summary, indent=2))


def _train_one_run(here, patient_id, graph_strategy, sop_minutes, device, compile_seconds):
    row = {
        "patient_id": patient_id,
        "graph_strategy": graph_strategy,
        "sop_minutes": sop_minutes,
    }
    try:
        folder = _ready_folder(here, patient_id, graph_strategy, sop_minutes)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        return row
    attempt = 0
    while attempt < 2:
        attempt += 1
        try:
            row.update(
                _train_attempt(
                    folder, graph_strategy, sop_minutes, device, compile_seconds
                )
            )
            row["status"] = "done"
            row["attempts"] = attempt
            return row
        except Exception as exc:
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc()
    row["status"] = "failed"
    row["attempts"] = attempt
    return row


def _train_attempt(folder, graph_strategy, sop_minutes, device, compile_seconds):
    import mamba_ssm

    if graph_strategy == "attention":
        model = load_attention_evobrain(device=device)
    else:
        model = load_evobrain(device=device)
    model = model.to(device)
    optimizer = make_optimizer(model)
    extra = {
        "train_model": "evobrain" if graph_strategy != "attention" else "evobrain attention",
        "sop_minutes": sop_minutes,
        "train_torch_version": torch.__version__,
        "command": "python train_sop.py",
        "seed": LOCKED["seed"],
        "dataset_accession": LOCKED["dataset_accession"],
        "dataset_version": LOCKED["dataset_version"],
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "cuda_version": torch.version.cuda,
        "mamba_ssm_version": mamba_ssm.__version__,
        "train_batch_size": LOCKED["train_batch_size"],
        "eval_batch_size": LOCKED["eval_batch_size"],
        "graph_strategy": graph_strategy,
    }
    _folder, _metrics = train_packed_eval_run(
        folder,
        model,
        optimizer,
        extra=extra,
        device=device,
        compile_seconds=compile_seconds,
        replace_existing_train=True,
    )
    _save_train_scores(folder, model, device)
    replace_metrics_keys(folder, _epoch_keys(folder))
    metrics = read_metrics(folder)
    return {
        "best_epoch": metrics.get("best_epoch"),
        "n_epochs_run": metrics.get("n_epochs_run"),
        "best_monitor_bce": metrics.get("best_monitor_bce"),
        "logit_zero_monitor_bce": metrics.get("logit_zero_monitor_bce"),
        "train_seconds": metrics.get("train_seconds"),
        "score_seconds": metrics.get("score_seconds"),
        "pos_weight": metrics.get("pos_weight"),
        "parameter_count": metrics.get("parameter_count"),
    }


def _ready_folder(here, patient_id, graph_strategy, sop_minutes):
    folder = run_folder_path(
        here, patient_id, "eval1", graph_strategy=graph_strategy, sop_minutes=sop_minutes
    )
    if graph_strategy == "attention" and not os.path.isfile(
        os.path.join(folder, "train.npz")
    ):
        pearson = run_folder_path(
            here, patient_id, "eval1", graph_strategy="pearson", sop_minutes=sop_minutes
        )
        return prepare_attention_folder(
            here, patient_id, pearson, sop_minutes=sop_minutes,
            command="python train_sop.py",
        )
    if not os.path.isfile(os.path.join(folder, "train.npz")):
        raise ValueError(f"pack is missing, pack this person first: {folder}")
    return folder


def _save_train_scores(folder, model, device, sensor_mask_fn=None):
    train_f, train_g, _train_y, train_ends = load_split(os.path.join(folder, "train.npz"))
    if train_ends is None:
        raise ValueError("train pack has no stored ends")
    if sensor_mask_fn is not None:
        train_f = sensor_mask_fn(train_f)
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


def _write_summary(here, summary, wall_start):
    summary["wall_seconds"] = float(time.perf_counter() - wall_start)
    out = os.path.join(here, SUMMARY_NAME)
    with open(out, "w") as handle:
        json.dump(summary, indent=2, fp=handle)
        handle.write("\n")


if __name__ == "__main__":
    main()
