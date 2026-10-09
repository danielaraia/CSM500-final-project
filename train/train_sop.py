import json
import os
import sys

import torch

from core.config import ALLOWED_SEEDS, LOCKED, run_folder_path
from core.randomness import seed_everything
from evaluation.alarms import save_window_scores
from evaluation.epoch_trace import install_logged_train_loop
from evaluation.metrics import read_metrics, replace_metrics_keys
from graphs.attention import load_attention_evobrain
from graphs.identity import load_identity_evobrain
from model.evobrain_load import load_evobrain
from model.train_loop import score_windows
from model.train_step import make_optimizer
from packs.pack import load_split
from packs.seed_pack import prepare_seed_folder
from run import train_packed_eval_run

PATIENTS = ("sub-004", "sub-015", "sub-039", "sub-051", "sub-052", "sub-073", "sub-113")
GRAPH_STRATEGIES = ("pearson", "mi", "attention")
TRAIN_GRAPHS = GRAPH_STRATEGIES + ("identity",)
NEW_SOP_MINUTES = (30, 15, 5)


def parse_args(argv):
    if len(argv) != 5:
        raise ValueError(
            "usage: python -m train.train_sop patient graph sop seed"
        )
    patient, graph = argv[1], argv[2]
    sop, seed = int(argv[3]), int(argv[4])
    if patient not in PATIENTS:
        raise ValueError(f"unknown patient: {patient}")
    if graph not in TRAIN_GRAPHS:
        raise ValueError(f"unknown graph: {graph}")
    if sop not in NEW_SOP_MINUTES:
        raise ValueError(f"unknown sop: {sop}")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"unknown seed: {seed}")
    return patient, graph, sop, seed


def ready_folder(root, patient, graph, sop, seed=123):
    folder = run_folder_path(root, patient, "eval1", graph, sop, seed)
    if os.path.isfile(os.path.join(folder, "train.npz")):
        return folder
    source_graph = "pearson" if graph in ("attention", "identity") else graph
    source = run_folder_path(root, patient, "eval1", source_graph, sop, 123)
    if folder == source:
        raise ValueError(f"pack is missing: {folder}")
    return prepare_seed_folder(root, patient, source, graph, sop, seed)


def make_model(graph, device):
    if graph == "attention":
        return load_attention_evobrain(device=device)
    if graph == "identity":
        return load_identity_evobrain(device=device)
    return load_evobrain(device=device)


def save_train_scores(folder, model, device, sensor_mask_fn=None):
    features, graphs, _labels, ends = load_split(
        os.path.join(folder, "train.npz")
    )
    if sensor_mask_fn is not None:
        features = sensor_mask_fn(features)
    scores = score_windows(model, features, graphs, device=device)
    save_window_scores(
        os.path.join(folder, "train_scores.npz"), ends, scores.numpy()
    )


def _epoch_keys(folder):
    with open(os.path.join(folder, "checkpoints", "epoch_log.json")) as handle:
        log = json.load(handle)
    return {
        "logit_zero_monitor_bce": log["logit_zero_monitor_bce"],
        "epoch_log": log["epochs"],
        "n_epochs_logged": log["n_epochs_logged"],
    }


_ready_folder = ready_folder
_save_train_scores = save_train_scores


def main():
    patient, graph, sop, seed = parse_args(sys.argv)
    if not torch.cuda.is_available():
        raise ValueError("a gpu is needed")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = ready_folder(root, patient, graph, sop, seed)
    metrics = read_metrics(folder)
    if metrics.get("train_seconds") is not None and os.path.isfile(
        os.path.join(folder, "test_scores.npz")
    ):
        print("already trained", folder)
        return

    import mamba_ssm

    seed_everything(seed)
    install_logged_train_loop()
    device = "cuda"
    model = make_model(graph, device).to(device)
    optimizer = make_optimizer(model)
    extra = {
        "train_model": "evobrain",
        "graph_strategy": graph,
        "sop_minutes": sop,
        "seed": seed,
        "command": "python -m train.train_sop",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mamba_ssm_version": mamba_ssm.__version__,
    }
    train_packed_eval_run(
        folder,
        model,
        optimizer,
        extra=extra,
        device=device,
        compile_seconds=0.0,
        replace_existing_train=True,
        seed=seed,
    )
    save_train_scores(folder, model, device)

    replace_metrics_keys(folder, _epoch_keys(folder))
    print("trained", folder)


if __name__ == "__main__":
    main()
