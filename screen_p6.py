"""Clock level screen for the wide screening cohort.
Writes p6-screening-cohort.json.
"""

import json
import math
import os
import re
import sys

from alarms import HOURS_TO_RESOLVE_FP_MAX, test_leads_from_seizures
from catch_coverage import catch_coverage_from_ends
from events import impd_spans, load_event_rows, seizure_spans
from glue import _keep_pack_ends
from labels import (
    LABEL_INTERICTAL,
    LABEL_SOP,
    label_second,
    seizures_with_full_sop,
    split_cut_times,
)
from pearson import steps_with_full_movement_buffer
from score_alarms import recording_duration_seconds
from windows import TRAIN_STRIDE_SECONDS, WINDOW_SECONDS, end_is_on_stride

META_ROOT = os.path.join("data", "seizeit2-meta")
OUT_NAME = "p6-screening-cohort.json"
SCREEN_SOP_SECONDS = (15 * 60, 10 * 60, 5 * 60)
# the split minimum from labels.n_test_seizures, which raises under 2: one earlier
# seizure trains, one later seizure tests. clock_screen.py used 3 for its SOP 30
# catch screen; here the exact split and monitor simulation decide packability.
MIN_USABLE_SEIZURES = 2
MIN_TEST_INTERICTAL_HOURS = HOURS_TO_RESOLVE_FP_MAX
RAM_GIB_SKIP = 14.0
COHORT_CAP = 20
BYTES_PER_WINDOW = WINDOW_SECONDS * 8 * 128 * 8 + WINDOW_SECONDS * 8 * 8 * 8
# already packed and trained in earlier work (sub-007 is unusable)
EXCLUDED_PATIENTS = (
    "sub-004",
    "sub-007",
    "sub-015",
    "sub-039",
    "sub-051",
    "sub-052",
    "sub-073",
    "sub-113",
)
RUN_RE = re.compile(r"(sub-\d+)_ses-01_task-szMonitoring_run-(\d+)_events\.tsv$")


def event_paths(meta_root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(meta_root):
        for name in filenames:
            if name.endswith("_events.tsv"):
                out.append(os.path.join(dirpath, name))
    out.sort()
    return out


def sidecar_skip(meta_root, patient_id, run_id):
    base = os.path.join(meta_root, patient_id, "ses-01")
    stem = f"{patient_id}_ses-01_task-szMonitoring_run-{run_id}"
    paths = {
        key: os.path.join(base, key, f"{stem}_{key}.json")
        for key in ("eeg", "ecg", "emg", "mov")
    }
    for key in ("eeg", "ecg", "emg", "mov"):
        if not os.path.isfile(paths[key]):
            return f"missing {key} json"
    with open(paths["mov"]) as handle:
        mov = json.load(handle)
    with open(paths["eeg"]) as handle:
        eeg = json.load(handle)
    if int(mov.get("MOVChannelCount", 0)) != 12:
        return f"MOVChannelCount {mov.get('MOVChannelCount')}"
    if int(eeg.get("EEGChannelCount", 0)) < 2:
        return f"EEGChannelCount {eeg.get('EEGChannelCount')}"
    return None


def graph_ok_seconds(duration, drop_spans):
    # same meaning as glue: a second is dropped when [s, s+1) overlaps a span
    quality = bytearray(b"\x01") * duration
    for start, end in drop_spans:
        first = max(0, int(math.floor(start - 1)) + 1)
        last = min(duration, int(math.ceil(end)))
        if first < last:
            quality[first:last] = b"\x00" * (last - first)
    has_buffer = steps_with_full_movement_buffer(list(quality))
    return [
        bool(quality[second] and has_buffer[second]) for second in range(duration)
    ]


def hop1_ends(graph_ok):
    # window_end_times(graph_ok, WINDOW_SECONDS, 1) via a prefix sum
    prefix = [0]
    for ok in graph_ok:
        prefix.append(prefix[-1] + (1 if ok else 0))
    ends = []
    end_time = WINDOW_SECONDS - 1
    while end_time < len(graph_ok):
        start_time = end_time - WINDOW_SECONDS + 1
        if prefix[end_time + 1] - prefix[start_time] == WINDOW_SECONDS:
            ends.append(end_time)
        end_time += 1
    return ends


def screen_sop(seizures, duration, packed_ends_graph_ok, sop_seconds):
    row = {"sop_minutes": sop_seconds // 60, "skip": None, "clock_pass": False}
    usable = seizures_with_full_sop(seizures, sop_seconds)
    row["n_usable"] = len(usable)
    if len(usable) < MIN_USABLE_SEIZURES:
        row["skip"] = f"under {MIN_USABLE_SEIZURES} usable, we got {len(usable)}"
        return row
    try:
        t1, t2, n_train, n_test = split_cut_times(usable, sop_seconds)
    except Exception as err:
        row["skip"] = f"cut {err}"
        return row
    row["t1"] = t1
    row["t2"] = t2
    row["n_train_seizures"] = n_train
    row["n_test_seizures"] = n_test
    kept = _keep_pack_ends(
        packed_ends_graph_ok, seizures, t1, TRAIN_STRIDE_SECONDS, sop_seconds
    )
    row["n_labelled_windows"] = len(kept)
    labels = [label_second(end, seizures, sop_seconds) for end in kept]
    train_pairs = [(e, lab) for e, lab in zip(kept, labels) if e < t1]
    val_pairs = [(e, lab) for e, lab in zip(kept, labels) if t1 <= e < t2]
    test_pairs = [(e, lab) for e, lab in zip(kept, labels) if e >= t2]
    hop10 = [e for e, _lab in train_pairs if end_is_on_stride(e, TRAIN_STRIDE_SECONDS)]
    n_monitor = len(hop10) // 4
    if n_monitor == 0:
        row["skip"] = "monitor empty"
        return row
    monitor_clock = hop10[len(hop10) - n_monitor]
    grad_labels = [lab for e, lab in train_pairs if e < monitor_clock]
    monitor_labels = [lab for e, lab in train_pairs if e >= monitor_clock]
    row["n_positive_train"] = grad_labels.count(LABEL_SOP)
    row["n_negative_train"] = grad_labels.count(LABEL_INTERICTAL)
    row["n_positive_monitor"] = monitor_labels.count(LABEL_SOP)
    row["n_negative_monitor"] = monitor_labels.count(LABEL_INTERICTAL)
    row["n_val_windows"] = len(val_pairs)
    row["n_test_windows"] = len(test_pairs)
    row["val_interictal_hours"] = round(
        [lab for _e, lab in val_pairs].count(LABEL_INTERICTAL) / 3600.0, 3
    )
    row["test_interictal_hours"] = round(
        [lab for _e, lab in test_pairs].count(LABEL_INTERICTAL) / 3600.0, 3
    )
    if row["n_positive_train"] < 1:
        row["skip"] = "train 0 SOP"
        return row
    if row["n_negative_train"] < 1:
        row["skip"] = "train 0 interictal"
        return row
    if row["n_positive_monitor"] < 1 or row["n_negative_monitor"] < 1:
        row["skip"] = (
            "monitor one class "
            f"pos={row['n_positive_monitor']} neg={row['n_negative_monitor']}"
        )
        return row
    if len(test_pairs) < 1:
        row["skip"] = "no test windows"
        return row
    scored = [int(e) for e, _lab in val_pairs] + [int(e) for e, _lab in test_pairs]
    try:
        _t1, _t2, test_leads, _n_merged, _n_test = test_leads_from_seizures(
            seizures, sop_seconds
        )
        coverage = catch_coverage_from_ends(scored, test_leads, duration, sop_seconds)
    except Exception as err:
        row["skip"] = f"catch {err}"
        return row
    for key in coverage:
        row[key] = coverage[key]
    if not coverage["catch_evaluable"]:
        row["skip"] = (
            "not catch evaluable, max lookback "
            f"{coverage['max_lookback_scores_in_hit']}"
        )
        return row
    if row["test_interictal_hours"] < MIN_TEST_INTERICTAL_HOURS:
        row["skip"] = (
            f"test interictal hours {row['test_interictal_hours']} "
            f"under {MIN_TEST_INTERICTAL_HOURS:.2f}"
        )
        return row
    row["clock_pass"] = True
    return row


def screen_run(meta_root, patient_id, run_id, events_path):
    row = {"patient_id": patient_id, "run_id": run_id, "skip": None}
    skip = sidecar_skip(meta_root, patient_id, run_id)
    if skip is not None:
        row["skip"] = skip
        return row
    try:
        event_rows = load_event_rows(events_path)
        seizures = seizure_spans(event_rows)
        duration = recording_duration_seconds(event_rows)
        drops = impd_spans(event_rows)
    except Exception as err:
        row["skip"] = f"events {err}"
        return row
    row["n_seizures"] = len(seizures)
    if len(seizures) < 1:
        row["skip"] = "no seizures"
        return row
    row["duration_seconds"] = duration
    graph_ok = graph_ok_seconds(duration, drops)
    ends = hop1_ends(graph_ok)
    row["per_sop"] = {}
    n_windows_max = 0
    for sop_seconds in SCREEN_SOP_SECONDS:
        sop_row = screen_sop(seizures, duration, ends, sop_seconds)
        row["per_sop"][str(sop_seconds // 60)] = sop_row
        n_windows_max = max(n_windows_max, sop_row.get("n_labelled_windows", 0))
    row["ram_gib"] = round(n_windows_max * BYTES_PER_WINDOW / (1024.0 ** 3), 2)
    row["needs_pod"] = bool(row["ram_gib"] >= RAM_GIB_SKIP)
    return row


def _run_rank_key(row):
    # longest passing sop first, then usable seizures, then clean hours, then ram
    for sop_seconds in SCREEN_SOP_SECONDS:
        sop_row = row["per_sop"][str(sop_seconds // 60)]
        if sop_row.get("clock_pass"):
            return (
                -sop_seconds,
                -sop_row["n_usable"],
                -sop_row["test_interictal_hours"],
                row["ram_gib"],
                row["run_id"],
            )
    return None


def choose_cohort(rows, cap=COHORT_CAP):
    by_patient = {}
    for row in rows:
        if row.get("skip") or row["patient_id"] in EXCLUDED_PATIENTS:
            continue
        by_patient.setdefault(row["patient_id"], []).append(row)
    candidates = []
    for patient_id, patient_rows in sorted(by_patient.items()):
        keyed = [(key, row) for row in patient_rows
                 if (key := _run_rank_key(row)) is not None]
        if not keyed:
            continue
        keyed.sort(key=lambda item: item[0])
        best_key, best_row = keyed[0]
        longest_sop_seconds = -best_key[0]
        sop_row = best_row["per_sop"][str(longest_sop_seconds // 60)]
        candidates.append({
            "patient_id": patient_id,
            "run_id": best_row["run_id"],
            "longest_pass_sop_minutes": longest_sop_seconds // 60,
            "n_usable_at_longest_sop": sop_row["n_usable"],
            "test_interictal_hours": sop_row["test_interictal_hours"],
            "val_interictal_hours": sop_row["val_interictal_hours"],
            "n_train_seizures": sop_row["n_train_seizures"],
            "n_test_seizures": sop_row["n_test_seizures"],
            "n_test_leads_evaluable": sop_row["n_test_leads_evaluable"],
            "ram_gib": best_row["ram_gib"],
            "needs_pod": best_row["needs_pod"],
            "duration_seconds": best_row["duration_seconds"],
            "passes": {
                sop: best_row["per_sop"][sop]["clock_pass"]
                for sop in best_row["per_sop"]
            },
        })
    candidates.sort(
        key=lambda c: (
            -c["n_usable_at_longest_sop"],
            -c["test_interictal_hours"],
            c["patient_id"],
        )
    )
    return candidates[:cap], candidates


def main():
    meta_root = META_ROOT
    rows = []
    n_event_files = 0
    for path in event_paths(meta_root):
        match = RUN_RE.search(path)
        if match is None:
            continue
        n_event_files += 1
        patient_id, run_id = match.group(1), match.group(2)
        row = screen_run(meta_root, patient_id, run_id, path)
        rows.append(row)
        if row.get("skip") is None:
            print(json.dumps({"screened": patient_id, "run": run_id}), flush=True)
    cohort, all_candidates = choose_cohort(rows)
    summary = {
        "command": "python screen_p6.py",
        "screen_sop_minutes": [s // 60 for s in SCREEN_SOP_SECONDS],
        "min_usable_seizures": MIN_USABLE_SEIZURES,
        "min_test_interictal_hours": MIN_TEST_INTERICTAL_HOURS,
        "ram_gib_skip": RAM_GIB_SKIP,
        "cohort_cap": COHORT_CAP,
        "excluded_patients": list(EXCLUDED_PATIENTS),
        "n_event_files": n_event_files,
        "n_runs_with_seizures": len([r for r in rows if r.get("skip") is None]),
        "n_patients_with_a_passing_run": len(all_candidates),
        "n_cohort": len(cohort),
        "cohort": cohort,
        "runs": rows,
    }
    with open(OUT_NAME, "w") as handle:
        json.dump(summary, indent=2, fp=handle)
        handle.write("\n")
    print(f"WROTE {OUT_NAME} n_cohort {len(cohort)} of {len(all_candidates)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
