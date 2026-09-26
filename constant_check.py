import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

from data.labels import label_second
from packs.pack import label_to_class_number
from tiny_model import TinyLogits
from train_step import make_optimizer, model_logits, one_train_step
from data.windows import WINDOW_SECONDS

DEAD_GRAPH_MAX_DIFF = 1e-12


def class_mix(labels):
    n_sop = 0
    n_interictal = 0
    flat = np.asarray(labels).reshape(-1)
    index = 0
    while index < flat.shape[0]:
        lab = int(flat[index])
        if lab == 1:
            n_sop += 1
        elif lab == 0:
            n_interictal += 1
        else:
            raise ValueError(f"class numbers have to be 0 or 1, we got {lab}")
        index += 1
    return {
        "n_sop": int(n_sop),
        "n_interictal": int(n_interictal),
        "mixed": n_sop > 0 and n_interictal > 0,
    }


def ends_time_ordered(ends):
    values = np.asarray(ends).reshape(-1)
    if values.shape[0] < 2:
        return True
    index = 1
    while index < values.shape[0]:
        if int(values[index]) <= int(values[index - 1]):
            return False
        index += 1
    return True


def labels_match_clock(ends, labels, seizures):
    ends = np.asarray(ends).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if ends.shape[0] != labels.shape[0]:
        raise ValueError(
            "ends and labels need the same number of windows, "
            f"we got {ends.shape[0]} and {labels.shape[0]}"
        )
    n_mismatch = 0
    index = 0
    while index < ends.shape[0]:
        want = label_to_class_number(label_second(int(ends[index]), seizures))
        if int(labels[index]) != want:
            n_mismatch += 1
        index += 1
    return int(n_mismatch)


def graph_spread(adjacency, n_sample=64):
    if getattr(adjacency, "ndim", None) != 4:
        raise ValueError(
            "graphs need 4 axes (windows, seconds, nodes, nodes), "
            f"we got {getattr(adjacency, 'ndim', None)}"
        )
    n_windows = adjacency.shape[0]
    if n_windows < 2:
        raise ValueError("need at least two windows to judge the graph")
    take = n_windows
    if take > n_sample:
        take = n_sample
    idx = _even_indices(n_windows, take)
    sample = np.asarray(adjacency[idx])
    g0 = sample[0]
    diffs = []
    n_identical = 0
    pos = 0
    while pos < take:
        mean_abs = float(np.mean(np.abs(sample[pos] - g0)))
        diffs.append(mean_abs)
        if mean_abs <= DEAD_GRAPH_MAX_DIFF:
            n_identical += 1
        pos += 1
    off = sample.copy()
    node = 0
    while node < off.shape[-1]:
        off[:, :, node, node] = 0.0
        node += 1
    return {
        "n_compared": int(take),
        "max_mean_abs_diff": float(max(diffs)),
        "median_mean_abs_diff": float(np.median(diffs)),
        "n_identical_to_first": int(n_identical),
        "off_diag_std": float(np.std(off)),
        "dead": max(diffs) <= DEAD_GRAPH_MAX_DIFF,
    }


def tiny_batch_learning(features, adjacency, labels, pos_weight, n_steps=5, seed=123):
    mix = class_mix(labels)
    if not mix["mixed"]:
        raise ValueError("tiny batch needs both SOP and interictal windows")
    if pos_weight <= 0:
        raise ValueError(f"pos_weight has to be > 0, we got {pos_weight}")
    torch.manual_seed(seed)
    model = TinyLogits()
    optimizer = make_optimizer(model)
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    g = torch.as_tensor(np.asarray(adjacency), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float32).reshape(-1)
    seq = torch.full((x.shape[0],), WINDOW_SECONDS)
    before = model.w.weight.detach().clone()
    losses = []
    step = 0
    while step < n_steps:
        loss = one_train_step(model, optimizer, x, seq, g, y, pos_weight)
        losses.append(float(loss))
        step += 1
    after = model.w.weight.detach()
    model.zero_grad()
    logits = model_logits(model, x, seq, g)
    recon = BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(pos_weight, dtype=logits.dtype)
    )(logits, y)
    recon.backward()
    grad_abs = float(model.w.weight.grad.abs().sum())
    return {
        "n_steps": int(n_steps),
        "losses": losses,
        "weight_moved": not torch.equal(before, after),
        "weight_delta": float((after - before).abs().max()),
        "grad_abs_sum": grad_abs,
        "grad_nonzero": grad_abs > 0.0,
    }


def checkpoint_health(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = blob["model"]
    n_params = 0
    n_nan = 0
    n_zero = 0
    n_zero_std = 0
    for _name, tensor in state.items():
        n_params += int(tensor.numel())
        values = tensor.float()
        if torch.isnan(values).any():
            n_nan += 1
        if int(torch.count_nonzero(tensor).item()) == 0:
            n_zero += 1
        if tensor.numel() > 1 and float(values.std()) == 0.0:
            n_zero_std += 1
    max_exp = 0.0
    opt = blob.get("optimizer")
    if isinstance(opt, dict):
        for _tid, row in opt.get("state", {}).items():
            if "exp_avg" in row:
                peak = float(row["exp_avg"].float().abs().max())
                if peak > max_exp:
                    max_exp = peak
    return {
        "epoch": int(blob.get("epoch", 0)),
        "n_tensors": int(len(state)),
        "n_params": int(n_params),
        "n_nan_tensors": int(n_nan),
        "n_all_zero_tensors": int(n_zero),
        "n_zero_std_tensors": int(n_zero_std),
        "optimizer_max_exp_avg": float(max_exp),
        "finite": n_nan == 0,
    }


def state_dicts_max_abs_diff(state_a, state_b):
    if len(state_a) != len(state_b):
        raise ValueError(
            "checkpoints need the same number of tensors, "
            f"we got {len(state_a)} and {len(state_b)}"
        )
    n_equal = 0
    max_abs = 0.0
    for name in state_a:
        if name not in state_b:
            raise ValueError(f"checkpoint is missing tensor {name}")
        a = state_a[name].float().reshape(-1)
        b = state_b[name].float().reshape(-1)
        if torch.equal(a, b):
            n_equal += 1
            continue
        diff = float((a - b).abs().max())
        if diff > max_abs:
            max_abs = diff
    return {"n_equal_tensors": int(n_equal), "n_tensors": int(len(state_a)), "max_abs_diff": float(max_abs)}


def _even_indices(n, take):
    if take >= n:
        return np.arange(n)
    if take < 2:
        return np.asarray([0])
    out = []
    pos = 0
    while pos < take:
        out.append(int(round(pos * (n - 1) / float(take - 1))))
        pos += 1
    return np.asarray(out)
