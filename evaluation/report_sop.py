import json
import os

from core.config import run_folder_path
from evaluation.metrics import read_metrics

# same list of runs as train_sop. it is written out again here so this file
# works without torch. train_sop.py loads torch at the top.
PATIENTS = ("sub-004", "sub-051", "sub-052", "sub-113", "sub-039", "sub-073", "sub-015")
GRAPH_STRATEGIES = ("pearson", "mi", "attention")
NEW_SOP_MINUTES = (15, 5)

GRID_NAME = "p4-sop-grid.json"

ROW_KEYS = (
    "best_epoch",
    "n_epochs_run",
    "best_monitor_bce",
    "pos_weight",
    "n_positive_train",
    "n_negative_train",
    "n_seizures_caught",
    "n_seizures_missed",
    "n_seizures_evaluable",
    "n_seizures_unevaluable",
    "event_sensitivity",
    "n_false_alarms",
    "false_alarms_per_hour",
    "scorable_interictal_hours",
    "resolves_0_15_per_hour",
    "threshold_set",
    "threshold_rule",
    "alarm_lookback_seconds",
    "alarm_vote_need",
    "study_criterion_met",
    "train_seconds",
    "score_seconds",
)


def grid_row(here, patient_id, graph_strategy, sop_minutes):
    folder = run_folder_path(
        here, patient_id, "eval1", graph_strategy=graph_strategy, sop_minutes=sop_minutes
    )
    row = {
        "patient_id": patient_id,
        "graph_strategy": graph_strategy,
        "sop_minutes": sop_minutes,
    }
    metrics_path = os.path.join(folder, "metrics.json")
    if not os.path.isfile(metrics_path):
        row["status"] = "missing"
        return row
    metrics = read_metrics(folder)
    if "train_seconds" not in metrics:
        row["status"] = "packed_not_trained"
        return row
    if "event_sensitivity" not in metrics and "n_seizures_caught" not in metrics:
        row["status"] = "trained_not_scored"
    else:
        row["status"] = "scored"
    for key in ROW_KEYS:
        if key in metrics:
            row[key] = metrics[key]
    return row


def build_grid(here):
    rows = []
    for sop_minutes in NEW_SOP_MINUTES:
        for patient_id in PATIENTS:
            for graph_strategy in GRAPH_STRATEGIES:
                rows.append(grid_row(here, patient_id, graph_strategy, sop_minutes))
    return rows


def plain_table(rows):
    head = (
        "patient | graph | sop | status | evaluable | caught | missed | "
        "unevaluable | fa | fa per hour | threshold"
    )
    lines = [head]
    for row in rows:
        if row["status"] != "scored":
            lines.append(
                f"{row['patient_id']} | {row['graph_strategy']} | {row['sop_minutes']} | "
                f"{row['status']} | | | | | | |"
            )
            continue
        fa_per_hour = row.get("false_alarms_per_hour")
        if fa_per_hour is None:
            fa_per_hour_text = "none"
        else:
            fa_per_hour_text = f"{fa_per_hour:.3f}"
        lines.append(
            f"{row['patient_id']} | {row['graph_strategy']} | {row['sop_minutes']} | "
            f"scored | {row.get('n_seizures_evaluable')} | "
            f"{row.get('n_seizures_caught')} | {row.get('n_seizures_missed')} | "
            f"{row.get('n_seizures_unevaluable')} | {row.get('n_false_alarms')} | "
            f"{fa_per_hour_text} | {row.get('threshold_set')}"
        )
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = build_grid(here)
    out = os.path.join(here, GRID_NAME)
    with open(out, "w") as handle:
        json.dump({"command": "python report_sop.py", "runs": rows}, indent=2, fp=handle)
        handle.write("\n")
    print(plain_table(rows))
    n_scored = 0
    for row in rows:
        if row["status"] == "scored":
            n_scored += 1
    print(f"scored {n_scored} of {len(rows)} runs")


if __name__ == "__main__":
    main()
