import json
import os

GATE_METRIC_KEYS = (
    "seconds_per_training_sample",
    "seconds_per_scored_second",
    "peak_memory_allocated_bytes",
    "concurrent_train_processes_arithmetic",
    "concurrent_train_processes_live",
    "batch_128_one_step_fits",
)


def load_gate_metrics(path):
    if not os.path.isfile(path):
        raise ValueError(f"p0 gate metrics file is missing: {path}")
    with open(path) as handle:
        gate = json.load(handle)
    extra = {}
    for key in GATE_METRIC_KEYS:
        if key not in gate:
            raise ValueError(f"p0 gate metrics missing {key}")
        extra[key] = gate[key]
    extra["timing_source"] = "p0-gate-metrics.json"
    return extra


def write_metrics(folder, metrics):
    path = os.path.join(folder, "metrics.json")
    if os.path.exists(path):
        raise ValueError(f"metrics already exist, we wont write over it: {path}")
    with open(path, "w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return path


def read_metrics(folder):
    path = os.path.join(folder, "metrics.json")
    if not os.path.isfile(path):
        raise ValueError(f"metrics file is missing: {path}")
    with open(path) as handle:
        return json.load(handle)
