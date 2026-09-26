import os

import torch
from torch.nn import BCEWithLogitsLoss

from core.config import LOCKED
from metrics import write_metrics
from pos_weight import weight_from_class_numbers
from train_step import model_logits, one_train_step
from windows import WINDOW_SECONDS


def monitor_bce(model, features, adjacency, labels, pos_weight, batch_size=None, device=None):
    if batch_size is None:
        batch_size = LOCKED["eval_batch_size"]
    if pos_weight <= 0:
        raise ValueError(f"pos_weight has to be > 0, we got {pos_weight}")
    features, adjacency, labels = _as_tensors(features, adjacency, labels, device)
    model.eval()
    total = 0.0
    n_windows = 0
    start = 0
    with torch.no_grad():
        while start < features.shape[0]:
            end = start + batch_size
            if end > features.shape[0]:
                end = features.shape[0]
            logits = model_logits(
                model,
                features[start:end],
                _seq_lengths(end - start, features.device),
                adjacency[start:end],
            )
            batch_labels = labels[start:end].reshape(-1)
            weight = torch.as_tensor(
                pos_weight, dtype=logits.dtype, device=logits.device
            )
            loss = BCEWithLogitsLoss(pos_weight=weight)(logits, batch_labels)
            n_batch = end - start
            total += float(loss) * n_batch
            n_windows += n_batch
            start = end
    if n_windows < 1:
        raise ValueError("monitor set is empty")
    return total / float(n_windows)


def score_windows(model, features, adjacency, batch_size=None, device=None):
    if batch_size is None:
        batch_size = LOCKED["eval_batch_size"]
    features = torch.as_tensor(features, dtype=torch.float32)
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    if features.shape[0] != adjacency.shape[0]:
        raise ValueError(
            "features and graphs need the same number of windows, "
            f"we got {features.shape[0]} and {adjacency.shape[0]}"
        )
    if features.shape[0] < 1:
        return torch.zeros(0)
    if device is not None:
        features = features.to(device)
        adjacency = adjacency.to(device)
    model.eval()
    chunks = []
    start = 0
    with torch.no_grad():
        while start < features.shape[0]:
            end = start + batch_size
            if end > features.shape[0]:
                end = features.shape[0]
            logits = model_logits(
                model,
                features[start:end],
                _seq_lengths(end - start, features.device),
                adjacency[start:end],
            )
            chunks.append(logits.detach().cpu())
            start = end
    if len(chunks) == 0:
        return torch.zeros(0)
    return torch.cat(chunks, dim=0)


def train_until_stop(
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
    best_loss = None
    best_epoch = 0
    wait = 0
    epoch = 0
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
    if checkpoint_dir is not None and best_epoch > 0:
        _load_checkpoint(os.path.join(checkpoint_dir, "best.pth.tar"), model)
    return best_epoch, best_loss, epoch


def train_and_write_metrics(
    folder,
    model,
    optimizer,
    train_features,
    train_adjacency,
    train_labels,
    monitor_features,
    monitor_adjacency,
    monitor_labels,
    extra=None,
    max_epochs=None,
    patience=None,
    train_batch_size=None,
    device=None,
):
    pos_weight, n_negative, n_positive = weight_from_class_numbers(train_labels)
    best_epoch, best_loss, n_epochs_run = train_until_stop(
        model,
        optimizer,
        train_features,
        train_adjacency,
        train_labels,
        monitor_features,
        monitor_adjacency,
        monitor_labels,
        pos_weight,
        max_epochs=max_epochs,
        patience=patience,
        train_batch_size=train_batch_size,
        checkpoint_dir=os.path.join(folder, "checkpoints"),
        device=device,
    )
    n_params = 0
    for parameter in model.parameters():
        n_params += parameter.numel()
    metrics = {
        "best_epoch": int(best_epoch),
        "n_epochs_run": int(n_epochs_run),
        "best_monitor_bce": float(best_loss),
        "pos_weight": float(pos_weight),
        "n_positive_train": int(n_positive),
        "n_negative_train": int(n_negative),
        "n_train_windows": int(len(train_labels)),
        "n_monitor_windows": int(len(monitor_labels)),
        "parameter_count": int(n_params),
        "negative_control": "none",
        "early_stopping_metric": LOCKED["early_stopping_metric"],
    }
    if extra is not None:
        for key in extra:
            metrics[key] = extra[key]
    write_metrics(folder, metrics)
    return best_epoch, best_loss


def _run_train_epoch(
    model,
    optimizer,
    features,
    adjacency,
    labels,
    pos_weight,
    batch_size,
    loss_fn=None,
    augment_fn=None,
):
    start = 0
    n_windows = features.shape[0]
    if n_windows < 1:
        raise ValueError("train set is empty")
    while start < n_windows:
        end = start + batch_size
        if end > n_windows:
            end = n_windows
        batch_features = features[start:end]
        if augment_fn is not None:
            # train slice only, and a fresh copy so the pack tensors never change
            batch_features = augment_fn(batch_features)
        one_train_step(
            model,
            optimizer,
            batch_features,
            _seq_lengths(end - start, features.device),
            adjacency[start:end],
            labels[start:end],
            pos_weight,
            loss_fn=loss_fn,
        )
        start = end


def _save_checkpoint(path, model, optimizer, epoch, monitor_loss):
    folder = os.path.dirname(path)
    if folder != "" and not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "monitor_loss": monitor_loss,
        },
        path,
    )


def _load_checkpoint(path, model):
    device = next(model.parameters()).device
    blob = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(blob["model"])
    return blob


def _as_tensors(features, adjacency, labels, device=None):
    features = torch.as_tensor(features, dtype=torch.float32)
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    if features.shape[0] != adjacency.shape[0] or features.shape[0] != labels.shape[0]:
        raise ValueError(
            "features, graphs and labels need the same number of windows, "
            f"we got {features.shape[0]}, {adjacency.shape[0]}, {labels.shape[0]}"
        )
    if device is not None:
        features = features.to(device)
        adjacency = adjacency.to(device)
        labels = labels.to(device)
    return features, adjacency, labels


def _seq_lengths(batch, device):
    return torch.full((batch,), WINDOW_SECONDS, device=device)
