import json
import os

# pearson is the default. mi is the other fixed graph this slice can pack.
ALLOWED_GRAPH_STRATEGIES = ("pearson", "mi")

# these beat the evobrain args.py defaults
LOCKED = {
    "dataset_accession": "ds005873",
    "dataset_version": "1.1.0",
    "seed": 123,
    "graph_strategy": "pearson",
    "sop_minutes": 30,
    "sph_minutes": 5,
    "interictal_distance_minutes": 60,
    "max_seq_len": 60,
    "input_dim": 128,
    "num_nodes": 8,
    "num_eigenvectors": 4,
    "top_k": 3,
    "learning_rate": 0.0001,
    "train_batch_size": 32,
    "eval_batch_size": 256,
    "train_stride_seconds": 10,
    "eval_stride_seconds": 1,
    "weight_decay": 0.0005,
    "dropout": 0.0,
    "max_epochs": 100,
    "patience": 5,
    "data_augment": False,
    "early_stopping_metric": "class_weighted_bce",
    "rnn_units": 64,
    "agg": "max",
    "num_classes": 1,
    "gnn": "ssg",
    "log_epsilon": 1e-8,
    "notch_hz": 50,
    "notch_q": 30,
    "magnitude": "sqrt(x*x + y*y + z*z)",
    "fft_bins": "drop dc, keep 1 to 128",
    "electrical_to_25hz": "scipy.signal.resample to 100 samples",
    "eeg_rms_rule": "drop a second only if both ears fail 13 to 150 uV",
    "filter_call": "scipy.signal.butter SOS then sosfiltfilt",
    "node_order": "eeg a, eeg b, ecg, emg, neck acc, neck gyr, chest acc, chest gyr",
    "early_stopping_direction": "minimise",
}


def write_config(path, extra):
    config = dict(LOCKED)
    for key in extra:
        if key == "graph_strategy":
            _require_graph_strategy(extra[key])
            config[key] = extra[key]
            continue
        if key in LOCKED and extra[key] != LOCKED[key]:
            raise ValueError(
                f"locked config key {key} is {LOCKED[key]!r}, extra tried {extra[key]!r}"
            )
        config[key] = extra[key]
    with open(path, "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def read_config(path):
    with open(path) as handle:
        return json.load(handle)


def run_folder_path(root, patient_id, leaf="none", graph_strategy=None):
    if leaf is None or leaf == "":
        raise ValueError("need a run leaf folder name")
    if graph_strategy is None:
        graph_strategy = LOCKED["graph_strategy"]
    _require_graph_strategy(graph_strategy)
    return os.path.join(
        root,
        "experiments",
        "runs",
        "within_patient",
        patient_id,
        graph_strategy,
        "sop30",
        "seed123",
        leaf,
    )


def start_run(root, patient_id, extra=None, leaf="none"):
    if patient_id is None or patient_id == "":
        raise ValueError("need a patient id")
    extra = dict(extra or {})
    graph_strategy = extra.get("graph_strategy", LOCKED["graph_strategy"])
    path = run_folder_path(root, patient_id, leaf, graph_strategy=graph_strategy)
    if os.path.exists(path):
        raise ValueError(f"run folder already exists, we wont write over it: {path}")
    os.makedirs(path)
    extra["patient_id"] = patient_id
    extra["run_leaf"] = leaf
    write_config(os.path.join(path, "config.json"), extra)
    return path


def _require_graph_strategy(graph_strategy):
    if graph_strategy not in ALLOWED_GRAPH_STRATEGIES:
        raise ValueError(
            "graph_strategy has to be pearson or mi, "
            f"we got {graph_strategy!r}"
        )
