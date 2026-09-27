import json
import os

from evaluation.logit_overlap import overlap_from_pack_and_scores, still_overlap
from evaluation.metrics import merge_alarm_metrics, read_metrics, replace_metrics_keys
from evaluation.score_alarms import alarm_metrics_from_folder


MI_FOLDERS = (
    "experiments/runs/within_patient/sub-051/mi/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-052/mi/sop30/seed123/eval1",
)

PEARSON_FOLDERS = (
    "experiments/runs/within_patient/sub-051/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
)

EVENTS_PATHS = {
    "sub-051": os.path.join(
        "data",
        "seizeit2",
        "sub-051",
        "ses-01",
        "eeg",
        "sub-051_ses-01_task-szMonitoring_run-01_events.tsv",
    ),
    "sub-052": os.path.join(
        "data",
        "seizeit2",
        "sub-052",
        "ses-01",
        "eeg",
        "sub-052_ses-01_task-szMonitoring_run-07_events.tsv",
    ),
}

SPLIT_NAMES = ("train", "monitor", "val", "test")

ROW_KEYS = (
    "patient_id",
    "graph_strategy",
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
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary = {"command": "python report_mi.py", "mi": [], "pearson": [], "table": []}
    mi_rows = []
    for rel in MI_FOLDERS:
        folder = os.path.join(here, rel)
        row = _score_folder(folder, here)
        mi_rows.append(row)
        summary["mi"].append(row)
        print("folder", folder)
        print(json.dumps(row, indent=2))
    pearson_rows = []
    for rel in PEARSON_FOLDERS:
        folder = os.path.join(here, rel)
        loaded = read_metrics(folder)
        row = _row_from_loaded(loaded, {}, loaded.get("logit_overlap") or {})
        pearson_rows.append(row)
        summary["pearson"].append(row)
        print("pearson", folder)
        print(json.dumps(row, indent=2))
    summary["table"] = _english_table(pearson_rows, mi_rows)
    out = os.path.join(here, "p2-mi-graph-summary.json")
    with open(out, "w") as handle:
        json.dump(summary, indent=2, fp=handle)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def _score_folder(folder, here):
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
    extra["graph_strategy"] = "mi"
    merge_alarm_metrics(folder, extra, replace=True)
    replace_metrics_keys(
        folder,
        {
            "logit_overlap": extra["logit_overlap"],
            "test_still_overlap": extra["test_still_overlap"],
            "train_still_overlap": extra["train_still_overlap"],
            "monitor_still_overlap": extra["monitor_still_overlap"],
            "false_alarms_unresolvable": extra["false_alarms_unresolvable"],
            "graph_strategy": "mi",
        },
    )
    loaded = read_metrics(folder)
    return _row_from_loaded(loaded, extra, overlap)


def _english_table(pearson_rows, mi_rows):
    by_person_pearson = {}
    for row in pearson_rows:
        by_person_pearson[row["patient_id"]] = row
    table = []
    for mi_row in mi_rows:
        patient_id = mi_row["patient_id"]
        if patient_id not in by_person_pearson:
            raise ValueError(f"no pearson row for {patient_id}")
        pearson_row = by_person_pearson[patient_id]
        table.append(
            {
                "patient_id": patient_id,
                "pearson": _short_event(pearson_row),
                "mi": _short_event(mi_row),
            }
        )
    return table


def _short_event(row):
    overlap = row.get("logit_overlap") or {}
    test = overlap.get("test") or {}
    return {
        "sens": row.get("event_sensitivity"),
        "false_alarms_per_hour": row.get("false_alarms_per_hour"),
        "caught": row.get("n_seizures_caught"),
        "missed": row.get("n_seizures_missed"),
        "evaluable": row.get("n_seizures_evaluable"),
        "unevaluable": row.get("n_seizures_unevaluable"),
        "test_still_overlap": row.get("test_still_overlap"),
        "test_overlap_frac": test.get("frac_sop_at_or_under_interictal_max"),
        "threshold_set": row.get("threshold_set"),
    }


def _row_from_loaded(loaded, extra, overlap):
    row = {}
    for key in ROW_KEYS:
        if key in extra:
            row[key] = extra[key]
        else:
            row[key] = loaded.get(key)
    row["logit_overlap"] = overlap
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
