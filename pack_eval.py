import json
import os
import sys
import time

from catch_coverage import catch_coverage_from_folder
from core.config import LOCKED
from metrics import write_metrics
from mi import build_mi_graph
from run import write_split_run

PATIENT_ID = "sub-113"
RUN_ID = "04"
LEAF = "eval1"
GRAPH_BUILDERS = {
    "pearson": None,
    "mi": build_mi_graph,
}


def seizeit2_run_paths(patient_id, run_id):
    base = os.path.join("data", "seizeit2", patient_id, "ses-01")
    stem = f"{patient_id}_ses-01_task-szMonitoring_run-{run_id}"
    return {
        "eeg": os.path.join(base, "eeg", f"{stem}_eeg.edf"),
        "ecg": os.path.join(base, "ecg", f"{stem}_ecg.edf"),
        "emg": os.path.join(base, "emg", f"{stem}_emg.edf"),
        "mov": os.path.join(base, "mov", f"{stem}_mov.edf"),
        "events": os.path.join(base, "eeg", f"{stem}_events.tsv"),
    }


def main():
    patient_id = PATIENT_ID
    run_id = RUN_ID
    graph_strategy = LOCKED["graph_strategy"]
    if len(sys.argv) == 1:
        pass
    elif len(sys.argv) == 3:
        patient_id = sys.argv[1]
        run_id = sys.argv[2]
    elif len(sys.argv) == 4:
        patient_id = sys.argv[1]
        run_id = sys.argv[2]
        graph_strategy = sys.argv[3]
    else:
        raise ValueError("usage: python pack_eval.py [patient_id run_id [graph_strategy]]")
    if graph_strategy == "attention":
        raise ValueError(
            "attention is learned at train time, do not pack it from edf"
        )
    if graph_strategy not in GRAPH_BUILDERS:
        raise ValueError(
            "graph_strategy has to be pearson or mi, "
            f"we got {graph_strategy!r}"
        )
    paths = seizeit2_run_paths(patient_id, run_id)
    extra = {
        "command": "python pack_eval.py",
        "pack_scope": "seizure bearing run only",
        "negative_control": "none",
        "packed_run": run_id,
        "graph_strategy": graph_strategy,
    }
    clock = time.perf_counter()
    folder, counts = write_split_run(
        ".",
        patient_id,
        paths["eeg"],
        paths["ecg"],
        paths["emg"],
        paths["mov"],
        paths["events"],
        extra=extra,
        leaf=LEAF,
        graph_builder=GRAPH_BUILDERS[graph_strategy],
    )
    metrics = {
        "patient_id": patient_id,
        "command": "python pack_eval.py",
        "pack_scope": "seizure bearing run only",
        "pack_seconds": float(time.perf_counter() - clock),
        "negative_control": "none",
        "packed_run": run_id,
        "graph_strategy": graph_strategy,
    }
    for key in counts:
        if key == "means" or key == "stds":
            continue
        metrics[key] = counts[key]
    coverage = catch_coverage_from_folder(folder, paths["events"])
    for key in coverage:
        metrics[key] = coverage[key]
    write_metrics(folder, metrics)
    print("folder", folder)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
