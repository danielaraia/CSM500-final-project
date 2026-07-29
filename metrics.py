import json
import os


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
