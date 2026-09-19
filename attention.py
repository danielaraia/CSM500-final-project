import os
import shutil
import sys

import torch
import torch.nn as nn

from config import LOCKED, start_run
from evobrain_load import evobrain_clone_path, load_evobrain
from metrics import read_metrics, write_metrics

# GraphLearner is from the EvoBrain clone, MIT licence,
# itself adapted from IDGL, https://github.com/hugochan/IDGL
ATTENTION_HIDDEN = 32
ATTENTION_METRIC = "self_attention"
ATTENTION_ADDED_PARAMETERS = 2 * LOCKED["input_dim"] * ATTENTION_HIDDEN
SPLIT_NAMES = ("train", "monitor", "val", "test")
PACK_KEYS_FROM_PEARSON = (
    "t1",
    "t2",
    "n_train_seizures",
    "n_test_seizures",
    "n_train_windows",
    "n_monitor_windows",
    "n_validation_windows",
    "n_test_windows",
    "packed_run",
    "pack_scope",
)


def make_graph_learner():
    GraphLearner = _graph_learner_class()
    return GraphLearner(
        input_size=LOCKED["input_dim"],
        hidden_size=ATTENTION_HIDDEN,
        num_nodes=LOCKED["num_nodes"],
        metric_type=ATTENTION_METRIC,
    )


def learned_adjacency(learner, input_seq):
    if input_seq.ndim != 4:
        raise ValueError(
            "features need shape batch, time, nodes, bins, "
            f"we got {tuple(input_seq.shape)}"
        )
    batch, time_steps, n_nodes, n_bins = input_seq.shape
    if n_nodes != LOCKED["num_nodes"] or n_bins != LOCKED["input_dim"]:
        raise ValueError(
            "features need "
            f"{LOCKED['num_nodes']} nodes and {LOCKED['input_dim']} bins, "
            f"we got {n_nodes} and {n_bins}"
        )
    context = input_seq.reshape(batch * time_steps, n_nodes, n_bins)
    weights = learner(context)
    return weights.reshape(batch, time_steps, n_nodes, n_nodes)


class AttentionBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.learner = make_graph_learner()

    def forward(self, input_seq, seq_lengths, adj):
        learned = learned_adjacency(self.learner, input_seq)
        return self.backbone(input_seq, seq_lengths, learned)


def load_attention_evobrain(device="cpu"):
    backbone = load_evobrain(device=device)
    return AttentionBackbone(backbone)


def prepare_attention_folder(
    root, patient_id, pearson_folder, sop_minutes=None, command="python train_attention.py"
):
    if not os.path.isdir(pearson_folder):
        raise ValueError(f"pearson pack folder is missing: {pearson_folder}")
    extra = {
        "command": command,
        "graph_strategy": "attention",
        "packed_adjacency": "unused, graph learner replaces it",
        "feature_pack_source": pearson_folder,
        "attention_hidden": ATTENTION_HIDDEN,
        "attention_metric": ATTENTION_METRIC,
    }
    if sop_minutes is not None:
        extra["sop_minutes"] = sop_minutes
    folder = start_run(root, patient_id, extra=extra, leaf="eval1")
    try:
        for name in SPLIT_NAMES:
            src = os.path.join(pearson_folder, name + ".npz")
            dst = os.path.join(folder, name + ".npz")
            if not os.path.isfile(src):
                raise ValueError(f"pearson split is missing: {src}")
            _link_or_copy(src, dst)
        pack_metrics = {
            "patient_id": patient_id,
            "command": command,
            "graph_strategy": "attention",
            "packed_adjacency": "unused, graph learner replaces it",
            "feature_pack_source": pearson_folder,
            "run_leaf": "eval1",
        }
        if sop_minutes is not None:
            pack_metrics["sop_minutes"] = sop_minutes
        pearson_metrics = read_metrics(pearson_folder)
        for key in PACK_KEYS_FROM_PEARSON:
            if key not in pearson_metrics:
                raise ValueError(f"pearson metrics missing {key}: {pearson_folder}")
            pack_metrics[key] = pearson_metrics[key]
        write_metrics(folder, pack_metrics)
    except Exception:
        shutil.rmtree(folder)
        raise
    return folder


def _graph_learner_class():
    clone = evobrain_clone_path()
    if clone not in sys.path:
        sys.path.insert(0, clone)
    from model.graph_learner import GraphLearner

    return GraphLearner


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
