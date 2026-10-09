import json
import os
import sys

from core.config import ALLOWED_SEEDS, run_folder_path
from evaluation.metrics import merge_alarm_metrics, read_metrics
from evaluation.score_alarms import alarm_metrics_from_folder
from packs.pack_eval import seizeit2_run_paths
from train.train_sop import NEW_SOP_MINUTES, PATIENTS, TRAIN_GRAPHS


def parse_args(argv):
    if len(argv) != 5:
        raise ValueError(
            "usage: python -m evaluation.score_sop patient graph sop seed"
        )
    patient, graph = argv[1], argv[2]
    sop, seed = int(argv[3]), int(argv[4])
    if patient not in PATIENTS:
        raise ValueError(f"unknown patient: {patient}")
    if graph not in TRAIN_GRAPHS and graph != "logistic":
        raise ValueError(f"unknown graph: {graph}")
    if sop not in NEW_SOP_MINUTES:
        raise ValueError(f"unknown sop: {sop}")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"unknown seed: {seed}")
    return patient, graph, sop, seed


def score_one_folder(root, patient, graph, sop, seed):
    folder = run_folder_path(root, patient, "eval1", graph, sop, seed)
    if not os.path.isfile(os.path.join(folder, "test_scores.npz")):
        raise ValueError(f"test scores are missing: {folder}")
    metrics = read_metrics(folder)
    events = seizeit2_run_paths(patient, metrics["packed_run"])["events"]
    result = alarm_metrics_from_folder(folder, events, sop_seconds=sop * 60)
    merge_alarm_metrics(folder, result, replace=True)
    return result


def main():
    patient, graph, sop, seed = parse_args(sys.argv)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = score_one_folder(root, patient, graph, sop, seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
