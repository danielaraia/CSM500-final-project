import json
import sys
import time

from catch_coverage import catch_coverage_from_folder
from core.config import ALLOWED_SOP_MINUTES
from metrics import write_metrics
from mi import build_mi_graph
from packs.pack_eval import seizeit2_run_paths
from run import write_split_run

LEAF = "eval1"
GRAPH_BUILDERS = {
    "pearson": None,
    "mi": build_mi_graph,
}


def parse_args(argv):
    if len(argv) != 5:
        raise ValueError(
            "usage: python pack_sop.py patient_id run_id graph_strategy sop_minutes"
        )
    patient_id = argv[1]
    run_id = argv[2]
    graph_strategy = argv[3]
    if graph_strategy == "attention":
        raise ValueError(
            "attention is learned at train time, do not pack it from edf"
        )
    if graph_strategy not in GRAPH_BUILDERS:
        raise ValueError(
            "graph_strategy has to be pearson or mi, "
            f"we got {graph_strategy!r}"
        )
    try:
        sop_minutes = int(argv[4])
    except ValueError:
        raise ValueError(f"sop_minutes has to be a whole number, we got {argv[4]!r}")
    if sop_minutes not in ALLOWED_SOP_MINUTES:
        raise ValueError(
            "sop_minutes has to be 30, 15, 10, or 5, "
            f"we got {sop_minutes!r}"
        )
    return patient_id, run_id, graph_strategy, sop_minutes


def main():
    patient_id, run_id, graph_strategy, sop_minutes = parse_args(sys.argv)
    paths = seizeit2_run_paths(patient_id, run_id)
    extra = {
        "command": "python pack_sop.py",
        "pack_scope": "seizure bearing run only",
        "negative_control": "none",
        "packed_run": run_id,
        "graph_strategy": graph_strategy,
        "sop_minutes": sop_minutes,
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
        sop_seconds=sop_minutes * 60,
    )
    metrics = {
        "patient_id": patient_id,
        "command": "python pack_sop.py",
        "pack_scope": "seizure bearing run only",
        "pack_seconds": float(time.perf_counter() - clock),
        "negative_control": "none",
        "packed_run": run_id,
        "graph_strategy": graph_strategy,
        "sop_minutes": sop_minutes,
    }
    for key in counts:
        if key == "means" or key == "stds":
            continue
        metrics[key] = counts[key]
    coverage = catch_coverage_from_folder(
        folder, paths["events"], sop_seconds=sop_minutes * 60
    )
    for key in coverage:
        metrics[key] = coverage[key]
    write_metrics(folder, metrics)
    print("folder", folder)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
