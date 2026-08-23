import json
import os

from logit_overlap import overlap_from_pack_and_scores, still_overlap
from metrics import merge_alarm_metrics, read_metrics, replace_metrics_keys
from score_alarms import alarm_metrics_from_folder


PACKED_FOLDERS = (
    "experiments/runs/within_patient/sub-073/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-039/pearson/sop30/seed123/eval1",
)

COMPARISON_FOLDERS = (
    "experiments/runs/within_patient/sub-051/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
)

EVENTS_PATHS = {
    "sub-073": os.path.join(
        "data",
        "seizeit2",
        "sub-073",
        "ses-01",
        "eeg",
        "sub-073_ses-01_task-szMonitoring_run-50_events.tsv",
    ),
    "sub-039": os.path.join(
        "data",
        "seizeit2",
        "sub-039",
        "ses-01",
        "eeg",
        "sub-039_ses-01_task-szMonitoring_run-06_events.tsv",
    ),
}

SPLIT_NAMES = ("train", "monitor", "val", "test")

ROW_KEYS = (
    "patient_id",
    "best_epoch",
    "n_epochs_run",
    "n_epochs_logged",
    "best_monitor_bce",
    "logit_zero_monitor_bce",
    "n_seizures_caught",
    "n_seizures_missed",
    "n_seizures_evaluable",
    "n_seizures_unevaluable",
    "event_sensitivity",
    "n_false_alarms",
    "false_alarms_per_hour",
    "false_alarms_unresolvable",
    "threshold_set",
    "alarm_threshold",
    "test_still_overlap",
    "train_still_overlap",
    "monitor_still_overlap",
)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    summary = {"command": "python report_073_039.py", "patients": [], "comparison": []}
    for rel in PACKED_FOLDERS:
        folder = os.path.join(here, rel)
        metrics = read_metrics(folder)
        patient_id = metrics["patient_id"]
        if patient_id not in EVENTS_PATHS:
            raise ValueError(f"no events path for {patient_id}")
        overlap = {}
        for name in SPLIT_NAMES:
            pack_path = os.path.join(folder, name + ".npz")
            scores_path = os.path.join(folder, name + "_scores.npz")
            if not os.path.isfile(pack_path):
                raise ValueError(f"pack is missing: {pack_path}")
            if not os.path.isfile(scores_path):
                raise ValueError(f"window scores are missing: {scores_path}")
            overlap[name] = overlap_from_pack_and_scores(pack_path, scores_path)
        extra = alarm_metrics_from_folder(
            folder, os.path.join(here, EVENTS_PATHS[patient_id])
        )
        extra["logit_overlap"] = overlap
        extra["test_still_overlap"] = _overlap_flag(overlap["test"])
        extra["train_still_overlap"] = _overlap_flag(overlap["train"])
        extra["monitor_still_overlap"] = _overlap_flag(overlap["monitor"])
        extra["false_alarms_unresolvable"] = _unresolvable(
            extra["scorable_interictal_hours"]
        )
        merge_alarm_metrics(folder, extra, replace=True)
        replace_metrics_keys(
            folder,
            {
                "logit_overlap": extra["logit_overlap"],
                "test_still_overlap": extra["test_still_overlap"],
                "train_still_overlap": extra["train_still_overlap"],
                "monitor_still_overlap": extra["monitor_still_overlap"],
                "false_alarms_unresolvable": extra["false_alarms_unresolvable"],
            },
        )
        loaded = read_metrics(folder)
        row = _row_from_loaded(loaded, extra, overlap)
        summary["patients"].append(row)
        print("folder", folder)
        print(json.dumps(row, indent=2))
    for rel in COMPARISON_FOLDERS:
        folder = os.path.join(here, rel)
        loaded = read_metrics(folder)
        row = _comparison_row(loaded)
        summary["comparison"].append(row)
        print("comparison", folder)
        print(json.dumps(row, indent=2))
    out = os.path.join(here, "p2-train-073-039-summary.json")
    with open(out, "w") as handle:
        json.dump(summary, indent=2, fp=handle)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def _row_from_loaded(loaded, extra, overlap):
    row = {}
    for key in ROW_KEYS:
        if key in extra:
            row[key] = extra[key]
        else:
            row[key] = loaded.get(key)
    row["logit_overlap"] = overlap
    return row


def _comparison_row(loaded):
    row = {}
    for key in ROW_KEYS:
        row[key] = loaded.get(key)
    row["logit_overlap"] = loaded.get("logit_overlap")
    return row


def _overlap_flag(row):
    if row["frac_sop_at_or_under_interictal_max"] is None:
        return None
    return still_overlap(row)


def _unresolvable(scorable_hours):
    if scorable_hours is None:
        return True
    return scorable_hours <= 0


if __name__ == "__main__":
    main()
