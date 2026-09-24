import os

import numpy as np
import torch

from config import LOCKED, run_folder_path
from pack import load_split
from pos_weight import weight_from_class_numbers
from train_loop import train_until_stop
from train_step import make_optimizer

# pretraining only ever reads train and monitor packs. monitor is cut from the
# train slice at pack time, so no patient val or test data is involved
POOL_SPLIT_NAMES = ("train", "monitor")


def find_pretrain_folders(root, patient_id, graph_strategy, sop_minutes):
    pool_root = os.path.join(root, "experiments", "runs", "within_patient")
    if not os.path.isdir(pool_root):
        raise ValueError(f"patient runs folder is missing: {pool_root}")
    # attention packs are hardlinks of the pearson packs and the model learns its own adjacency, so the pearson pool is the same pool
    pool_strategy = graph_strategy
    if graph_strategy == "attention":
        pool_strategy = "pearson"
    folders = []
    for other in sorted(os.listdir(pool_root)):
        if other == patient_id:
            continue  # the target patient never pretrains on itself
        folder = run_folder_path(
            root,
            other,
            "eval1",
            graph_strategy=pool_strategy,
            sop_minutes=sop_minutes,
        )
        if os.path.isfile(os.path.join(folder, "train.npz")) and os.path.isfile(
            os.path.join(folder, "monitor.npz")
        ):
            folders.append(folder)
    return folders


def pool_splits(folders):
    if len(folders) < 1:
        raise ValueError("pretrain pool is empty, pack some other patients first")
    parts = {}
    for name in POOL_SPLIT_NAMES:
        parts[name] = []
    for folder in folders:
        for name in POOL_SPLIT_NAMES:
            parts[name].append(load_split(os.path.join(folder, name + ".npz")))
    train = _concat(parts["train"])
    monitor = _concat(parts["monitor"])
    return train[0], train[1], train[2], monitor[0], monitor[1], monitor[2]


def pretrain_model(
    root,
    model,
    patient_id,
    graph_strategy,
    sop_minutes,
    checkpoint_dir,
    loss_fn=None,
    augment_fn=None,
    sensor_mask_fn=None,
    device=None,
    max_epochs=None,
    patience=None,
    train_batch_size=None,
):
    folders = find_pretrain_folders(root, patient_id, graph_strategy, sop_minutes)
    train_f, train_g, train_y, monitor_f, monitor_g, monitor_y = pool_splits(folders)
    if sensor_mask_fn is not None:
        # the pool sees the same dropped nodes as the target patient
        train_f = sensor_mask_fn(train_f)
        monitor_f = sensor_mask_fn(monitor_f)
    torch.manual_seed(LOCKED["seed"])
    pos_weight, n_negative, n_positive = weight_from_class_numbers(train_y)
    optimizer = make_optimizer(model)
    best_epoch, best_loss, n_epochs_run = train_until_stop(
        model,
        optimizer,
        train_f,
        train_g,
        train_y,
        monitor_f,
        monitor_g,
        monitor_y,
        pos_weight,
        max_epochs=max_epochs,
        patience=patience,
        train_batch_size=train_batch_size,
        checkpoint_dir=checkpoint_dir,
        device=device,
        loss_fn=loss_fn,
        augment_fn=augment_fn,
    )
    return {
        "pretrain_pool": [_patient_from_folder(folder) for folder in folders],
        "pretrain_pool_size": int(len(folders)),
        "pretrain_n_train_windows": int(len(train_y)),
        "pretrain_n_monitor_windows": int(len(monitor_y)),
        "pretrain_n_positive_train": int(n_positive),
        "pretrain_n_negative_train": int(n_negative),
        "pretrain_pos_weight": float(pos_weight),
        "pretrain_best_epoch": int(best_epoch),
        "pretrain_best_monitor_bce": float(best_loss),
        "pretrain_n_epochs_run": int(n_epochs_run),
    }


def load_pretrain_weights(model, checkpoint_dir):
    path = os.path.join(checkpoint_dir, "best.pth.tar")
    if not os.path.isfile(path):
        raise ValueError(f"pretrain checkpoint is missing: {path}")
    device = next(model.parameters()).device
    blob = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(blob["model"])
    return int(blob["epoch"])


def _concat(splits):
    features = np.concatenate([split[0] for split in splits])
    adjacency = np.concatenate([split[1] for split in splits])
    labels = np.concatenate([split[2] for split in splits])
    return features, adjacency, labels


def _patient_from_folder(folder):
    # .../within_patient/<patient>/<graph>/sop<n>/seed123/eval1
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(folder))))
    return os.path.basename(path)
