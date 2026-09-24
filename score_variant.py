import json
import os
import sys
from datetime import datetime, timezone

from config import read_config, run_folder_path
from metrics import merge_alarm_metrics, read_metrics
from pack_eval import seizeit2_run_paths
from score_alarms import alarm_metrics_from_folder
from sensor_mask import SENSOR_TIERS

USAGE = (
    "usage: python score_variant.py patient_id graph_strategy sop_minutes run_leaf "
    "[--sensors full|headset|eeg_acc|eeg_only]"
)


def parse_args(argv):
    if len(argv) < 5:
        raise ValueError(USAGE)
    patient_id = argv[1]
    graph_strategy = argv[2]
    try:
        sop_minutes = int(argv[3])
    except ValueError:
        raise ValueError(f"sop_minutes has to be a whole number, we got {argv[3]!r}")
    leaf = argv[4]
    sensors = None
    rest = list(argv[5:])
    while len(rest) > 0:
        flag = rest.pop(0)
        if flag != "--sensors":
            raise ValueError(f"unknown flag {flag!r}, " + USAGE)
        if len(rest) < 1:
            raise ValueError("flag --sensors needs a value, " + USAGE)
        if sensors is not None:
            raise ValueError("flag --sensors given twice, " + USAGE)
        sensors = rest.pop(0)
    if sensors is None:
        sensors = "full"
    if sensors not in SENSOR_TIERS:
        raise ValueError(
            "sensors has to be full, headset, eeg_acc, or eeg_only, "
            f"we got {sensors!r}"
        )
    return patient_id, graph_strategy, sop_minutes, leaf, sensors


def score_variant_folder(
    here, patient_id, graph_strategy, sop_minutes, leaf, sensors="full"
):
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
    packed_sensors = cfg.get("sensors", "full")
    if packed_sensors != sensors:
        raise ValueError(
            "config sensors do not match the request, "
            f"config {packed_sensors!r} request {sensors!r} in {folder}"
        )
    if sensors != "full":
        row["sensors"] = sensors
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
    patient_id, graph_strategy, sop_minutes, leaf, sensors = parse_args(sys.argv)
    here = os.path.dirname(os.path.abspath(__file__))
    row = score_variant_folder(
        here, patient_id, graph_strategy, sop_minutes, leaf, sensors=sensors
    )
    print(json.dumps(row))


if __name__ == "__main__":
    main()
