import os
import sys
from types import SimpleNamespace

from config import LOCKED

# EvoBrain clone, MIT licence, https://github.com/Kotoge/EvoBrain


def locked_model_args():
    return SimpleNamespace(
        num_nodes=LOCKED["num_nodes"],
        input_dim=LOCKED["input_dim"],
        rnn_units=LOCKED["rnn_units"],
        dropout=LOCKED["dropout"],
        agg=LOCKED["agg"],
        max_seq_len=LOCKED["max_seq_len"],
        top_k=LOCKED["top_k"],
        data_augment=LOCKED["data_augment"],
    )


def evobrain_clone_path():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "EvoBrain")
    if not os.path.isdir(path):
        raise ValueError("EvoBrain clone is missing, put it in seizure-tgnn/EvoBrain")
    return path


def load_evobrain(device="cpu"):
    missing = _missing_evobrain_packages()
    if len(missing) > 0:
        raise ValueError(
            "cant load EvoBrain, missing " + ", ".join(missing)
        )
    clone = evobrain_clone_path()
    if clone not in sys.path:
        sys.path.insert(0, clone)
    from model.EvoBrain import EvoBrain_classification

    args = locked_model_args()
    model = EvoBrain_classification(
        args,
        num_classes=LOCKED["num_classes"],
        device=device,
        gnn=LOCKED["gnn"],
        num_eigenvectors=LOCKED["num_eigenvectors"],
    )
    return model


def _missing_evobrain_packages():
    missing = []
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        missing.append("torch_geometric")
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        missing.append("mamba_ssm")
    return missing
