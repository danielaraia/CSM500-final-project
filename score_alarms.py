import json
import math
import os

from alarms import (
    FP_MAX_PER_HOUR,
    choose_operating_threshold,
    load_window_scores,
    logit_to_probability,
    score_at_threshold,
    score_map,
    study_criterion_met,
    test_leads_from_seizures,
)
from events import load_event_rows, seizure_spans
from metrics import merge_alarm_metrics, read_metrics
from pack import load_split


KNOWN_EVENTS = {
    "sub-004": os.path.join(
        "data",
        "seizeit2",
        "sub-004",
        "ses-01",
        "eeg",
        "sub-004_ses-01_task-szMonitoring_run-01_events.tsv",
    ),
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
    "sub-113": os.path.join(
        "data",
        "seizeit2",
        "sub-113",
        "ses-01",
        "eeg",
        "sub-113_ses-01_task-szMonitoring_run-04_events.tsv",
    ),
}

PACKED_FOLDERS = (
    "experiments/runs/within_patient/sub-004/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-051/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-113/pearson/sop30/seed123/eval1",
)


def recording_duration_seconds(rows):
    if len(rows) < 1:
        raise ValueError("events file is empty")
    raw = rows[0].get("recordingDuration")
    if raw is None or str(raw).strip() == "":
        raise ValueError("events file has no recordingDuration")
    duration = float(raw)
    if duration <= 0:
        raise ValueError(f"recordingDuration has to be > 0, we got {duration}")
    return int(math.ceil(duration))


def alarm_metrics_from_folder(folder, events_path):
    metrics = read_metrics(folder)
    val_ends, val_logits = load_window_scores(os.path.join(folder, "val_scores.npz"))
    test_ends, test_logits = load_window_scores(os.path.join(folder, "test_scores.npz"))
    score_at = score_map(val_ends, val_logits)
    test_map = score_map(test_ends, test_logits)
    for second in test_map:
        if second in score_at:
            raise ValueError(f"val and test both scored second {second}")
        score_at[second] = test_map[second]
    rows = load_event_rows(events_path)
    seizures = seizure_spans(rows)
    duration_seconds = recording_duration_seconds(rows)
    t1, t2, test_leads, n_merged, n_test = test_leads_from_seizures(seizures)
    if abs(float(metrics["t1"]) - t1) > 1e-6 or abs(float(metrics["t2"]) - t2) > 1e-6:
        raise ValueError(
            "cut times in metrics do not match the events file, "
            f"metrics t1={metrics['t1']} t2={metrics['t2']} events t1={t1} t2={t2}"
        )
    if int(metrics["n_test_seizures"]) != n_test:
        raise ValueError(
            "n_test_seizures in metrics does not match the events file, "
            f"metrics {metrics['n_test_seizures']} events {n_test}"
        )
    monitor_pack_path = os.path.join(folder, "monitor.npz")
    monitor_path = os.path.join(folder, "monitor_scores.npz")
    fallback_start = None
    fallback_end = None
    n_monitor_windows_with_scores = 0
    monitor_logits_loaded = False
    pack_ends = None
    if os.path.isfile(monitor_pack_path):
        pack_features, pack_graphs, pack_labels, pack_ends = load_split(
            monitor_pack_path
        )
        del pack_features, pack_graphs, pack_labels
        if pack_ends is None:
            raise ValueError("monitor pack has no stored ends")
        if len(pack_ends) > 0:
            fallback_start = int(min(pack_ends))
            fallback_end = int(max(pack_ends)) + 1
    if os.path.isfile(monitor_path) and pack_ends is not None:
        monitor_ends, monitor_logits = load_window_scores(monitor_path)
        if len(pack_ends) != len(monitor_ends):
            raise ValueError(
                "monitor scores and monitor pack need the same number of windows, "
                f"we got {len(monitor_ends)} and {len(pack_ends)}"
            )
        index = 0
        while index < len(monitor_ends):
            if int(pack_ends[index]) != int(monitor_ends[index]):
                raise ValueError(
                    "monitor score ends do not match the pack, "
                    f"pack {int(pack_ends[index])} scores {int(monitor_ends[index])}"
                )
            index += 1
        monitor_map = score_map(monitor_ends, monitor_logits)
        for second in monitor_map:
            if second in score_at:
                raise ValueError(f"monitor and val or test both scored second {second}")
            score_at[second] = monitor_map[second]
        n_monitor_windows_with_scores = int(len(monitor_ends))
        monitor_logits_loaded = True
    elif pack_ends is not None and len(pack_ends) > 0:
        index = 0
        while index < len(pack_ends):
            second = int(pack_ends[index])
            if second in score_at:
                raise ValueError(f"monitor and val or test both scored second {second}")
            # 0 is only a placeholder so scorable hours can be counted from ends
            score_at[second] = 0.0
            index += 1
    threshold, val_info = choose_operating_threshold(
        score_at,
        seizures,
        duration_seconds,
        t1,
        t2,
        fallback_start=fallback_start,
        fallback_end=fallback_end,
    )
    if (
        (not monitor_logits_loaded)
        and val_info["threshold_set"] == "monitor"
        and val_info["val_scorable_interictal_hours"] is not None
        and val_info["val_scorable_interictal_hours"] > 0
    ):
        raise ValueError(
            "monitor has scorable hours but window scores are missing, cannot pick a threshold"
        )
    test_scored = score_at_threshold(
        score_at,
        threshold,
        seizures,
        test_leads,
        duration_seconds,
        t2,
        duration_seconds,
    )
    if math.isinf(threshold):
        threshold_probability = None
    else:
        threshold_probability = logit_to_probability(threshold)
    extra = {
        "event_sensitivity": test_scored["event_sensitivity"],
        "n_seizures_caught": test_scored["n_seizures_caught"],
        "n_seizures_missed": test_scored["n_seizures_missed"],
        "n_seizures_evaluable": test_scored["n_seizures_evaluable"],
        "n_seizures_unevaluable": test_scored["n_seizures_unevaluable"],
        "n_test_lead_seizures": test_scored["n_seizures"],
        "n_seizures_merged": n_merged,
        "n_false_alarms": test_scored["n_false_alarms"],
        "false_alarms_per_hour": test_scored["false_alarms_per_hour"],
        "false_alarms_per_day": test_scored["false_alarms_per_day"],
        "raw_interictal_hours": test_scored["raw_interictal_hours"],
        "scorable_interictal_hours": test_scored["scorable_interictal_hours"],
        "interictal_hours_dropped": test_scored["interictal_hours_dropped"],
        "time_in_warning_interictal_hours": test_scored[
            "time_in_warning_interictal_hours"
        ],
        "alarm_threshold": None if math.isinf(threshold) else float(threshold),
        "alarm_threshold_probability": threshold_probability,
        "threshold_rule": val_info["threshold_rule"],
        "threshold_set": val_info["threshold_set"],
        "val_cap_met": val_info["val_cap_met"],
        "val_false_alarms_per_hour": val_info["val_false_alarms_per_hour"],
        "val_scorable_interictal_hours": val_info["val_scorable_interictal_hours"],
        "resolves_0_15_per_hour": test_scored["resolves_0_15_per_hour"],
        "study_criterion_met": study_criterion_met(
            test_scored["event_sensitivity"],
            test_scored["false_alarms_per_hour"],
            test_scored["n_false_alarms"],
            test_scored["resolves_0_15_per_hour"],
        ),
        "fp_max_per_hour": FP_MAX_PER_HOUR,
        "alarm_lookback_seconds": 300,
        "alarm_vote_need": 240,
        "alarm_silent_seconds": 2100,
        "n_test_windows_with_scores": int(len(test_ends)),
        "n_val_windows_with_scores": int(len(val_ends)),
        "n_monitor_windows_with_scores": n_monitor_windows_with_scores,
        "monitor_logits_loaded": monitor_logits_loaded,
    }
    return extra


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    summary = {"command": "python score_alarms.py", "patients": []}
    for rel in PACKED_FOLDERS:
        folder = os.path.join(here, rel)
        metrics = read_metrics(folder)
        patient_id = metrics["patient_id"]
        if patient_id not in KNOWN_EVENTS:
            raise ValueError(f"no events path for {patient_id}")
        extra = alarm_metrics_from_folder(
            folder, os.path.join(here, KNOWN_EVENTS[patient_id])
        )
        merge_alarm_metrics(folder, extra, replace=True)
        row = {
            "patient_id": patient_id,
            "n_seizures_caught": extra["n_seizures_caught"],
            "n_seizures_missed": extra["n_seizures_missed"],
            "n_seizures_evaluable": extra["n_seizures_evaluable"],
            "n_seizures_unevaluable": extra["n_seizures_unevaluable"],
            "event_sensitivity": extra["event_sensitivity"],
            "n_false_alarms": extra["n_false_alarms"],
            "false_alarms_per_hour": extra["false_alarms_per_hour"],
            "false_alarms_per_day": extra["false_alarms_per_day"],
            "scorable_interictal_hours": extra["scorable_interictal_hours"],
            "threshold_rule": extra["threshold_rule"],
            "threshold_set": extra["threshold_set"],
            "study_criterion_met": extra["study_criterion_met"],
        }
        summary["patients"].append(row)
        print("folder", folder)
        print(json.dumps(row, indent=2))
    out = os.path.join(here, "p2-alarm-metrics-summary.json")
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
