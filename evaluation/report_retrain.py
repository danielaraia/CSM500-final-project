import json
import os

from evaluation.logit_overlap import overlap_from_pack_and_scores, still_overlap
from evaluation.metrics import merge_alarm_metrics, read_metrics, replace_metrics_keys
from evaluation.score_alarms import KNOWN_EVENTS, alarm_metrics_from_folder


PACKED_FOLDERS = (
    "experiments/runs/within_patient/sub-051/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
)

SPLIT_NAMES = ("train", "monitor", "val", "test")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary = {"command": "python report_retrain.py", "patients": []}
    for rel in PACKED_FOLDERS:
        folder = os.path.join(here, rel)
        metrics = read_metrics(folder)
        patient_id = metrics["patient_id"]
        if patient_id not in KNOWN_EVENTS:
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
            folder, os.path.join(here, KNOWN_EVENTS[patient_id])
        )
        extra["logit_overlap"] = overlap
        extra["test_still_overlap"] = _overlap_flag(overlap["test"])
        extra["train_still_overlap"] = _overlap_flag(overlap["train"])
        extra["monitor_still_overlap"] = _overlap_flag(overlap["monitor"])
        merge_alarm_metrics(folder, extra, replace=True)
        replace_metrics_keys(
            folder,
            {
                "logit_overlap": extra["logit_overlap"],
                "test_still_overlap": extra["test_still_overlap"],
                "train_still_overlap": extra["train_still_overlap"],
                "monitor_still_overlap": extra["monitor_still_overlap"],
            },
        )
        loaded = read_metrics(folder)
        row = {
            "patient_id": patient_id,
            "best_epoch": loaded.get("best_epoch"),
            "n_epochs_run": loaded.get("n_epochs_run"),
            "n_epochs_logged": loaded.get("n_epochs_logged"),
            "best_monitor_bce": loaded.get("best_monitor_bce"),
            "logit_zero_monitor_bce": loaded.get("logit_zero_monitor_bce"),
            "n_seizures_caught": extra["n_seizures_caught"],
            "n_seizures_missed": extra["n_seizures_missed"],
            "n_seizures_evaluable": extra["n_seizures_evaluable"],
            "n_seizures_unevaluable": extra["n_seizures_unevaluable"],
            "event_sensitivity": extra["event_sensitivity"],
            "n_false_alarms": extra["n_false_alarms"],
            "false_alarms_per_hour": extra["false_alarms_per_hour"],
            "threshold_set": extra["threshold_set"],
            "alarm_threshold": extra["alarm_threshold"],
            "test_still_overlap": extra["test_still_overlap"],
            "train_still_overlap": extra["train_still_overlap"],
            "monitor_still_overlap": extra["monitor_still_overlap"],
            "logit_overlap": overlap,
        }
        summary["patients"].append(row)
        print("folder", folder)
        print(json.dumps(row, indent=2))
    out = os.path.join(here, "p2-honest-retrain-summary.json")
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def _overlap_flag(row):
    if row["frac_sop_at_or_under_interictal_max"] is None:
        return None
    return still_overlap(row)


if __name__ == "__main__":
    main()
