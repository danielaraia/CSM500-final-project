import json
import math
import os

import torch

from config import LOCKED
from train_loop import (
    _as_tensors,
    _load_checkpoint,
    _run_train_epoch,
    _save_checkpoint,
    monitor_bce,
)


def logit_zero_weighted_bce(n_positive, n_negative, pos_weight):
    n_windows = n_positive + n_negative
    if n_windows < 1:
        raise ValueError("need at least one window for the logit 0 loss")
    if pos_weight <= 0:
        raise ValueError(f"pos_weight has to be > 0, we got {pos_weight}")
    return math.log(2.0) * (pos_weight * n_positive + n_negative) / float(n_windows)


def class_counts(labels):
    n_positive = 0
    n_negative = 0
    flat = torch.as_tensor(labels, dtype=torch.float32).reshape(-1)
    index = 0
    while index < flat.shape[0]:
        lab = int(flat[index].item())
        if lab == 1:
            n_positive += 1
        elif lab == 0:
            n_negative += 1
        else:
            raise ValueError(f"class numbers have to be 0 or 1, we got {lab}")
        index += 1
    return n_positive, n_negative


def train_until_stop_logged(
    model,
    optimizer,
    train_features,
    train_adjacency,
    train_labels,
    monitor_features,
    monitor_adjacency,
    monitor_labels,
    pos_weight,
    max_epochs=None,
    patience=None,
    train_batch_size=None,
    checkpoint_dir=None,
    device=None,
    loss_fn=None,
    augment_fn=None,
):
    if max_epochs is None:
        max_epochs = LOCKED["max_epochs"]
    if patience is None:
        patience = LOCKED["patience"]
    if train_batch_size is None:
        train_batch_size = LOCKED["train_batch_size"]
    if max_epochs < 1:
        raise ValueError(f"max_epochs has to be at least 1, we got {max_epochs}")
    if patience < 1:
        raise ValueError(f"patience has to be at least 1, we got {patience}")
    train_features, train_adjacency, train_labels = _as_tensors(
        train_features, train_adjacency, train_labels, device
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=LOCKED["max_epochs"]
    )
    n_positive_monitor, n_negative_monitor = class_counts(monitor_labels)
    logit0 = logit_zero_weighted_bce(
        n_positive_monitor, n_negative_monitor, pos_weight
    )
    best_loss = None
    best_epoch = 0
    wait = 0
    epoch = 0
    rows = []
    while epoch < max_epochs:
        _run_train_epoch(
            model,
            optimizer,
            train_features,
            train_adjacency,
            train_labels,
            pos_weight,
            train_batch_size,
            loss_fn=loss_fn,
            augment_fn=augment_fn,
        )
        scheduler.step()
        train_loss = monitor_bce(
            model,
            train_features,
            train_adjacency,
            train_labels,
            pos_weight,
            device=device,
        )
        this_loss = monitor_bce(
            model,
            monitor_features,
            monitor_adjacency,
            monitor_labels,
            pos_weight,
            device=device,
        )
        epoch += 1
        improved = best_loss is None or this_loss < best_loss
        if improved:
            best_loss = this_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        rows.append(
            {
                "epoch": int(epoch),
                "train_bce": float(train_loss),
                "monitor_bce": float(this_loss),
                "improved": bool(improved),
                "wait": int(wait),
                "best_epoch": int(best_epoch),
            }
        )
        if checkpoint_dir is not None:
            _save_checkpoint(
                os.path.join(checkpoint_dir, "last.pth.tar"),
                model,
                optimizer,
                epoch,
                this_loss,
            )
            if improved:
                _save_checkpoint(
                    os.path.join(checkpoint_dir, "best.pth.tar"),
                    model,
                    optimizer,
                    epoch,
                    this_loss,
                )
        if wait >= patience:
            break
    if checkpoint_dir is not None:
        write_epoch_log(
            os.path.join(checkpoint_dir, "epoch_log.json"),
            rows,
            logit0,
            pos_weight,
            n_positive_monitor,
            n_negative_monitor,
        )
        if best_epoch > 0:
            _load_checkpoint(os.path.join(checkpoint_dir, "best.pth.tar"), model)
    return best_epoch, best_loss, epoch


def write_epoch_log(
    path, rows, logit_zero, pos_weight, n_positive_monitor, n_negative_monitor
):
    folder = os.path.dirname(path)
    if folder != "" and not os.path.isdir(folder):
        os.makedirs(folder)
    blob = {
        "logit_zero_monitor_bce": float(logit_zero),
        "pos_weight": float(pos_weight),
        "n_positive_monitor": int(n_positive_monitor),
        "n_negative_monitor": int(n_negative_monitor),
        "n_epochs_logged": int(len(rows)),
        "epochs": rows,
    }
    with open(path, "w") as handle:
        json.dump(blob, handle, indent=2)
        handle.write("\n")
    return path


def install_logged_train_loop():
    import run

    run.train_until_stop = train_until_stop_logged
