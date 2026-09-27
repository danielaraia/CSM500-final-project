import math
import os

import numpy as np

from evaluation.alarms import (
    FP_MAX_PER_HOUR,
    logit_to_probability,
    score_at_threshold,
    score_map,
    study_criterion_met,
    test_leads_from_seizures,
)
from data.events import load_event_rows, seizure_spans
from evaluation.logit_overlap import overlap_row, still_overlap
from evaluation.metrics import read_metrics
from model.pos_weight import weight_from_class_numbers
from evaluation.score_alarms import KNOWN_EVENTS, recording_duration_seconds
from evaluation.simple_baseline import (
    apply_zscore,
    fit_logistic,
    node_mean_energy,
    pick_threshold_from_val_or_monitor,
    predict_logits,
    threshold_score_map,
    zscore_from_train,
)


SPLIT_NAMES = ("train", "monitor", "val", "test")


def run_folder(folder, repo_root):
    metrics = read_metrics(folder)
    patient_id = metrics["patient_id"]
    events_rel = KNOWN_EVENTS.get(patient_id)
    if events_rel is None:
        raise ValueError(f"no events path for {patient_id}")
    rows = load_event_rows(os.path.join(repo_root, events_rel))
    seizures = seizure_spans(rows)
    duration_seconds = recording_duration_seconds(rows)
    t1, t2, test_leads, n_merged, n_test = test_leads_from_seizures(seizures)
    if abs(float(metrics["t1"]) - t1) > 1e-6 or abs(float(metrics["t2"]) - t2) > 1e-6:
        raise ValueError("cut times in metrics do not match the events file")
    if int(metrics["n_test_seizures"]) != n_test:
        raise ValueError("n_test_seizures in metrics does not match the events file")
    splits = {}
    name_index = 0
    while name_index < len(SPLIT_NAMES):
        name = SPLIT_NAMES[name_index]
        path = os.path.join(folder, name + ".npz")
        print("load", patient_id, name)
        splits[name] = load_strength_clock(path)
        name_index += 1
    train = splits["train"]
    pos_weight, _n_neg, _n_pos = weight_from_class_numbers(train["labels"])
    print("fit", patient_id)
    mean, std = zscore_from_train(train["strength"])
    weights = fit_logistic(
        apply_zscore(train["strength"], mean, std),
        train["labels"],
        pos_weight,
    )
    logits = {}
    overlap = {}
    name_index = 0
    while name_index < len(SPLIT_NAMES):
        name = SPLIT_NAMES[name_index]
        split = splits[name]
        logits[name] = predict_logits(
            apply_zscore(split["strength"], mean, std),
            weights,
        )
        overlap[name] = overlap_row(split["labels"], logits[name])
        name_index += 1
    print("threshold", patient_id)
    threshold, val_info = pick_threshold_from_val_or_monitor(
        splits["val"]["ends"],
        logits["val"],
        splits["monitor"]["ends"],
        logits["monitor"],
        seizures,
        duration_seconds,
        t1,
        t2,
    )
    score_at = threshold_score_map(
        splits["val"]["ends"],
        logits["val"],
        splits["monitor"]["ends"],
        logits["monitor"],
    )
    test_map = score_map(splits["test"]["ends"], logits["test"])
    for second in test_map:
        if second in score_at:
            raise ValueError(f"test and val or monitor both scored second {second}")
        score_at[second] = test_map[second]
    print("score test events", patient_id)
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
        threshold_value = None
        threshold_probability = None
    else:
        threshold_value = float(threshold)
        threshold_probability = logit_to_probability(threshold)
    baseline = {
        "event_sensitivity": test_scored["event_sensitivity"],
        "n_seizures_caught": test_scored["n_seizures_caught"],
        "n_seizures_missed": test_scored["n_seizures_missed"],
        "n_seizures_evaluable": test_scored["n_seizures_evaluable"],
        "n_seizures_unevaluable": test_scored["n_seizures_unevaluable"],
        "n_false_alarms": test_scored["n_false_alarms"],
        "false_alarms_per_hour": test_scored["false_alarms_per_hour"],
        "scorable_interictal_hours": test_scored["scorable_interictal_hours"],
        "alarm_threshold": threshold_value,
        "alarm_threshold_probability": threshold_probability,
        "threshold_rule": val_info["threshold_rule"],
        "threshold_set": val_info["threshold_set"],
        "val_cap_met": val_info["val_cap_met"],
        "study_criterion_met": study_criterion_met(
            test_scored["event_sensitivity"],
            test_scored["false_alarms_per_hour"],
            test_scored["n_false_alarms"],
            test_scored["resolves_0_15_per_hour"],
        ),
        "fp_max_per_hour": FP_MAX_PER_HOUR,
        "alarm_vote_need": 240,
        "pos_weight": float(pos_weight),
        "logit_overlap": overlap,
        "test_still_overlap": _overlap_flag(overlap["test"]),
        "train_still_overlap": _overlap_flag(overlap["train"]),
        "monitor_still_overlap": _overlap_flag(overlap["monitor"]),
    }
    return {
        "patient_id": patient_id,
        "n_seizures_merged": n_merged,
        "baseline": baseline,
        "evobrain": evo_from_metrics(metrics),
    }


def load_strength_clock(path):
    pack = np.load(path, mmap_mode="r")
    if "ends" not in pack.files:
        raise ValueError(f"pack has no stored ends: {path}")
    strength = node_mean_energy(pack["features"])
    labels = np.asarray(pack["labels"]).reshape(-1)
    ends = np.asarray(pack["ends"]).reshape(-1)
    if strength.shape[0] != labels.shape[0] or strength.shape[0] != ends.shape[0]:
        raise ValueError(
            "features, labels and ends need the same number of windows, "
            f"we got {strength.shape[0]}, {labels.shape[0]}, {ends.shape[0]}"
        )
    return {"strength": strength, "labels": labels, "ends": ends}


def evo_from_metrics(metrics):
    return {
        "event_sensitivity": metrics.get("event_sensitivity"),
        "n_seizures_caught": metrics.get("n_seizures_caught"),
        "n_seizures_missed": metrics.get("n_seizures_missed"),
        "n_seizures_evaluable": metrics.get("n_seizures_evaluable"),
        "n_seizures_unevaluable": metrics.get("n_seizures_unevaluable"),
        "n_false_alarms": metrics.get("n_false_alarms"),
        "false_alarms_per_hour": metrics.get("false_alarms_per_hour"),
        "threshold_set": metrics.get("threshold_set"),
        "test_still_overlap": metrics.get("test_still_overlap"),
        "train_still_overlap": metrics.get("train_still_overlap"),
        "study_criterion_met": metrics.get("study_criterion_met"),
        "logit_overlap": metrics.get("logit_overlap"),
    }


def _overlap_flag(row):
    if row["frac_sop_at_or_under_interictal_max"] is None:
        return None
    return still_overlap(row)
