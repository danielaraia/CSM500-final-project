import json
import os
import sys
from datetime import datetime, timezone

from config import read_config, run_folder_path
from metrics import merge_alarm_metrics, read_metrics
from pack_eval import seizeit2_run_paths
from score_alarms import alarm_metrics_from_folder


def score_variant_folder(here, patient_id, graph_strategy, sop_minutes, leaf):
    folder = run_folder_path(
        here, patient_id, leaf, graph_strategy=graph_strategy, sop_minutes=sop_minutes
    )
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": "p6-levers",
        "patient_id": patient_id,
        "graph_strategy": graph_strategy,
        "sop_minutes": sop_minutes,
        "run_leaf": leaf,
        "run_folder": os.path.relpath(folder, here),
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
            "config sop does not match the request, "
            f"config {packed_sop!r} request {sop_minutes!r} in {folder}"
        )
    row["recipe"] = cfg.get("recipe")
    row["loss"] = cfg.get("loss")
    row["augmentation"] = cfg.get("augmentation")
    row["pretrain"] = cfg.get("pretrain")
    row["seed"] = cfg.get("seed")
    metrics = read_metrics(folder)
    events_path = seizeit2_run_paths(patient_id, metrics["packed_run"])["events"]
    extra = alarm_metrics_from_folder(
        folder, events_path, sop_seconds=sop_minutes * 60
    )
    merge_alarm_metrics(folder, extra, replace=True)
    row["status"] = "done"
    row["n_seizures_evaluable"] = extra["n_seizures_evaluable"]
    row["n_seizures_caught"] = extra["n_seizures_caught"]
    row["n_seizures_missed"] = extra["n_seizures_missed"]
    row["n_seizures_unevaluable"] = extra["n_seizures_unevaluable"]
    row["val_fah"] = extra["val_false_alarms_per_hour"]
    row["test_fah"] = extra["false_alarms_per_hour"]
    row["n_false_alarms"] = extra["n_false_alarms"]
    row["threshold"] = extra["alarm_threshold"]
    row["threshold_set"] = extra["threshold_set"]
    row["scorable_interictal_hours"] = extra["scorable_interictal_hours"]
    return row


def main():
    if len(sys.argv) != 5:
        raise ValueError(
            "usage: python score_variant.py patient_id graph_strategy sop_minutes run_leaf"
        )
    here = os.path.dirname(os.path.abspath(__file__))
    row = score_variant_folder(here, sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    print(json.dumps(row))


if __name__ == "__main__":
    main()
