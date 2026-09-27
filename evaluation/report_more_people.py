import json
import os
import sys

from evaluation.logit_overlap import overlap_from_pack_and_scores, still_overlap
from evaluation.metrics import merge_alarm_metrics, read_metrics, replace_metrics_keys
from evaluation.score_alarms import alarm_metrics_from_folder


COMPARISON_FOLDERS = (
    "experiments/runs/within_patient/sub-051/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-039/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-073/pearson/sop30/seed123/eval1",
)

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


def packed_folder(patient_id):
    return os.path.join(
        "experiments",
        "runs",
        "within_patient",
        patient_id,
        "pearson",
        "sop30",
        "seed123",
        "eval1",
    )


def events_relpath(patient_id, packed_run):
    run_id = str(packed_run)
    stem = f"{patient_id}_ses-01_task-szMonitoring_run-{run_id}"
    return os.path.join(
        "data",
        "seizeit2-meta",
        patient_id,
        "ses-01",
        "eeg",
        f"{stem}_events.tsv",
    )


def resolve_events_path(here, patient_id, packed_run):
    meta = os.path.join(here, events_relpath(patient_id, packed_run))
    live = os.path.join(
        here,
        "data",
        "seizeit2",
        patient_id,
        "ses-01",
        "eeg",
        f"{patient_id}_ses-01_task-szMonitoring_run-{packed_run}_events.tsv",
    )
    if os.path.isfile(meta):
        return meta
    if os.path.isfile(live):
        return live
    raise ValueError(f"no events tsv for {patient_id} run {packed_run}")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patient_ids = _patient_ids(sys.argv)
    summary = {"command": "python report_more_people.py", "patients": [], "comparison": []}
    index = 0
    while index < len(patient_ids):
        rel = packed_folder(patient_ids[index])
        folder = os.path.join(here, rel)
        metrics = read_metrics(folder)
        patient_id = metrics["patient_id"]
        packed_run = metrics.get("packed_run")
        if packed_run is None:
            raise ValueError(f"metrics has no packed_run: {folder}")
        events_path = resolve_events_path(here, patient_id, packed_run)
        overlap = {}
        name_i = 0
        while name_i < len(SPLIT_NAMES):
            name = SPLIT_NAMES[name_i]
            pack_path = os.path.join(folder, name + ".npz")
            scores_path = os.path.join(folder, name + "_scores.npz")
            if not os.path.isfile(pack_path):
                raise ValueError(f"pack is missing: {pack_path}")
            if not os.path.isfile(scores_path):
                raise ValueError(f"window scores are missing: {scores_path}")
            overlap[name] = overlap_from_pack_and_scores(pack_path, scores_path)
            name_i += 1
        extra = alarm_metrics_from_folder(folder, events_path)
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
        index += 1
    cmp_i = 0
    while cmp_i < len(COMPARISON_FOLDERS):
        folder = os.path.join(here, COMPARISON_FOLDERS[cmp_i])
        loaded = read_metrics(folder)
        row = _comparison_row(loaded)
        summary["comparison"].append(row)
        print("comparison", folder)
        print(json.dumps(row, indent=2))
        cmp_i += 1
    out = os.path.join(here, "p2-more-people-summary.json")
    with open(out, "w") as handle:
        json.dump(summary, indent=2, fp=handle)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def _patient_ids(argv):
    if len(argv) < 2:
        raise ValueError("usage: python report_more_people.py sub-XXX [sub-YYY ...]")
    return tuple(argv[1:])


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
