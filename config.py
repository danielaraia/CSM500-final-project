import json

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
    "weight_decay": 0.0005,
    "dropout": 0.0,
    "max_epochs": 100,
    "patience": 5,
    "data_augment": False,
    "early_stopping_metric": "class_weighted_bce",
    "log_epsilon": 1e-8,
    "notch_hz": 50,
    "notch_q": 30,
    "magnitude": "sqrt(x*x + y*y + z*z)",
    "fft_bins": "drop dc, keep 1 to 128",
    "electrical_to_25hz": "scipy.signal.resample to 100 samples",
}


def write_config(path, extra):
    config = dict(LOCKED)
    for key in extra:
        config[key] = extra[key]
    with open(path, "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def read_config(path):
    with open(path) as handle:
        return json.load(handle)
