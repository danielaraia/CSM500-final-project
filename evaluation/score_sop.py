import json
import os
import sys

from core.config import read_config, run_folder_path
from evaluation.metrics import merge_alarm_metrics, read_metrics
from packs.pack_eval import seizeit2_run_paths
from evaluation.score_alarms import alarm_metrics_from_folder
from train.train_sop import GRAPH_STRATEGIES, NEW_SOP_MINUTES, PATIENTS

SUMMARY_NAME = "results/p4-sop-scores-summary.json"


def run_grid():
    grid = []
    for sop_minutes in NEW_SOP_MINUTES:
        for patient_id in PATIENTS:
            for graph_strategy in GRAPH_STRATEGIES:
                grid.append((patient_id, graph_strategy, sop_minutes))
    return grid


def score_one_folder(here, patient_id, graph_strategy, sop_minutes):
    folder = run_folder_path(
        here, patient_id, "eval1", graph_strategy=graph_strategy, sop_minutes=sop_minutes
    )
    row = {
        "patient_id": patient_id,
        "graph_strategy": graph_strategy,
        "sop_minutes": sop_minutes,
    }
    if not os.path.isdir(folder):
        row["status"] = "missing"
        row["error"] = f"run folder is missing: {folder}"
        return row
    if not os.path.isfile(os.path.join(folder, "test_scores.npz")):
        row["status"] = "missing"
        row["error"] = f"test scores are missing, train first: {folder}"
        return row
    cfg = read_config(os.path.join(folder, "config.json"))
    packed_sop = cfg.get("sop_minutes")
    if packed_sop != sop_minutes:
        raise ValueError(
            "config sop does not match the grid, "
            f"config {packed_sop!r} grid {sop_minutes!r} in {folder}"
        )
    metrics = read_metrics(folder)
    events_path = seizeit2_run_paths(patient_id, metrics["packed_run"])["events"]
    extra = alarm_metrics_from_folder(
        folder, events_path, sop_seconds=sop_minutes * 60
    )
    merge_alarm_metrics(folder, extra, replace=True)
    row["status"] = "done"
    for key in (
        "n_seizures_caught",
        "n_seizures_missed",
        "n_seizures_evaluable",
        "n_seizures_unevaluable",
        "event_sensitivity",
        "n_false_alarms",
        "false_alarms_per_hour",
        "scorable_interictal_hours",
        "threshold_rule",
        "threshold_set",
        "study_criterion_met",
    ):
        row[key] = extra[key]
    return row


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) == 1:
        grid = run_grid()
    elif len(sys.argv) == 4:
        grid = [(sys.argv[1], sys.argv[2], int(sys.argv[3]))]
    else:
        raise ValueError(
            "usage: python score_sop.py [patient_id graph_strategy sop_minutes]"
        )
    summary = {"command": "python score_sop.py", "runs": []}
    for patient_id, graph_strategy, sop_minutes in grid:
        row = score_one_folder(here, patient_id, graph_strategy, sop_minutes)
        summary["runs"].append(row)
        print(json.dumps(row, indent=2))
    out = os.path.join(here, SUMMARY_NAME)
    with open(out, "w") as handle:
        json.dump(summary, indent=2, fp=handle)
        handle.write("\n")
    n_failed = 0
    for row in summary["runs"]:
        if row["status"] != "done":
            n_failed += 1
    print(f"scored {len(summary['runs']) - n_failed} of {len(summary['runs'])} runs")


if __name__ == "__main__":
    main()
