import json
import os
import re
import sys

from evaluation.alarms import test_leads_from_seizures
from evaluation.catch_coverage import catch_coverage_from_ends
from data.events import impd_spans, load_event_rows, seizure_spans
from model.glue import _keep_pack_ends
from data.labels import (
    LABEL_INTERICTAL,
    LABEL_SOP,
    label_second,
    seizures_with_full_sop,
    split_cut_times,
)
from graphs.pearson import MOVEMENT_BUFFER_SECONDS
from evaluation.score_alarms import recording_duration_seconds
from data.windows import WINDOW_SECONDS, window_end_times

META_ROOT = os.path.join("data", "seizeit2-meta")
BYTES_PER_WINDOW = WINDOW_SECONDS * 8 * 128 * 8 + WINDOW_SECONDS * 8 * 8 * 8
RAM_GIB_SKIP = 14.0
ALREADY = {
    "sub-004",
    "sub-007",
    "sub-039",
    "sub-051",
    "sub-052",
    "sub-073",
    "sub-113",
}
SOP30 = {
    "sub-001",
    "sub-002",
    "sub-003",
    "sub-004",
    "sub-005",
    "sub-006",
    "sub-007",
    "sub-008",
    "sub-009",
    "sub-011",
    "sub-012",
    "sub-015",
    "sub-017",
    "sub-020",
    "sub-021",
    "sub-022",
    "sub-024",
    "sub-025",
    "sub-026",
    "sub-027",
    "sub-029",
    "sub-030",
    "sub-033",
    "sub-034",
    "sub-035",
    "sub-038",
    "sub-039",
    "sub-040",
    "sub-042",
    "sub-043",
    "sub-044",
    "sub-046",
    "sub-047",
    "sub-050",
    "sub-051",
    "sub-052",
    "sub-053",
    "sub-055",
    "sub-056",
    "sub-057",
    "sub-059",
    "sub-060",
    "sub-061",
    "sub-062",
    "sub-063",
    "sub-064",
    "sub-065",
    "sub-066",
    "sub-068",
    "sub-069",
    "sub-070",
    "sub-071",
    "sub-072",
    "sub-073",
    "sub-074",
    "sub-075",
    "sub-076",
    "sub-077",
    "sub-078",
    "sub-079",
    "sub-080",
    "sub-099",
    "sub-101",
    "sub-102",
    "sub-103",
    "sub-104",
    "sub-105",
    "sub-106",
    "sub-107",
    "sub-108",
    "sub-110",
    "sub-111",
    "sub-112",
    "sub-113",
    "sub-114",
    "sub-115",
    "sub-116",
    "sub-117",
    "sub-118",
    "sub-119",
}
RUN_RE = re.compile(r"(sub-\d+)_ses-01_task-szMonitoring_run-(\d+)_events\.tsv$")


def main():
    rows = []
    for events_path in _event_paths():
        match = RUN_RE.search(events_path)
        if match is None:
            continue
        patient_id = match.group(1)
        run_id = match.group(2)
        if patient_id not in SOP30:
            continue
        row = screen_run(patient_id, run_id, events_path)
        rows.append(row)
        print(json.dumps(row), flush=True)
    passes = []
    index = 0
    while index < len(rows):
        row = rows[index]
        if row.get("clock_pass") and row.get("catch_evaluable"):
            if row["patient_id"] not in ALREADY:
                passes.append(row)
        index += 1
    passes.sort(key=lambda item: (item["ram_gib"], item["patient_id"], item["run_id"]))
    ram_skipped = []
    index = 0
    while index < len(rows):
        row = rows[index]
        skip = row.get("skip") or ""
        if skip.startswith("ram ") and row.get("catch_evaluable"):
            if row["patient_id"] not in ALREADY:
                ram_skipped.append(row)
        index += 1
    ram_skipped.sort(key=lambda item: (item["ram_gib"], item["patient_id"], item["run_id"]))
    summary = {
        "n_runs_screened": len(rows),
        "n_new_clock_evaluable": len(passes),
        "new_clock_evaluable": passes,
        "n_new_ram_skipped_catch_evaluable": len(ram_skipped),
        "new_ram_skipped_catch_evaluable": ram_skipped,
    }
    out = os.path.join("results/p2-more-people-clock.json")
    with open(out, "w") as handle:
        json.dump(summary, indent=2, fp=handle)
        handle.write("\n")
    print("WROTE", out, "n_new", len(passes), flush=True)


def screen_run(patient_id, run_id, events_path):
    row = {
        "patient_id": patient_id,
        "run_id": run_id,
        "clock_pass": False,
        "skip": None,
    }
    sidecars = _sidecar_paths(patient_id, run_id)
    for key in ("eeg", "ecg", "emg", "mov"):
        if not os.path.isfile(sidecars[key]):
            row["skip"] = f"missing {key} json"
            return row
    mov = _load_json(sidecars["mov"])
    eeg = _load_json(sidecars["eeg"])
    if int(mov.get("MOVChannelCount", 0)) != 12:
        row["skip"] = f"MOVChannelCount {mov.get('MOVChannelCount')}"
        return row
    if int(eeg.get("EEGChannelCount", 0)) < 2:
        row["skip"] = f"EEGChannelCount {eeg.get('EEGChannelCount')}"
        return row
    try:
        event_rows = load_event_rows(events_path)
        seizures = seizure_spans(event_rows)
        duration = recording_duration_seconds(event_rows)
        usable = seizures_with_full_sop(seizures)
    except Exception as err:
        row["skip"] = f"events {err}"
        return row
    row["n_usable_sop30"] = len(usable)
    if len(usable) < 3:
        row["skip"] = f"under 3 usable SOP 30, we got {len(usable)}"
        return row
    try:
        t1, t2, n_train, n_test = split_cut_times(usable)
    except Exception as err:
        row["skip"] = f"cut {err}"
        return row
    row["t1"] = t1
    row["t2"] = t2
    row["n_train_seizures"] = n_train
    row["n_test_seizures"] = n_test
    quality_ok = _quality_ok(duration, impd_spans(event_rows))
    graph_ok = _graph_ok(quality_ok)
    hop1 = window_end_times(graph_ok, WINDOW_SECONDS, 1)
    kept = _keep_pack_ends(hop1, seizures, t1, 10)
    n_windows = len(kept)
    ram_gib = (n_windows * BYTES_PER_WINDOW) / (1024.0 ** 3)
    row["n_labelled_windows"] = n_windows
    row["ram_gib"] = round(ram_gib, 2)
    ram_over = ram_gib >= RAM_GIB_SKIP
    train_ends = []
    val_ends = []
    test_ends = []
    train_labels = []
    val_labels = []
    test_labels = []
    index = 0
    while index < len(kept):
        end_time = kept[index]
        lab = label_second(end_time, seizures)
        if end_time < t1:
            train_ends.append(end_time)
            train_labels.append(lab)
        elif end_time < t2:
            val_ends.append(end_time)
            val_labels.append(lab)
        else:
            test_ends.append(end_time)
            test_labels.append(lab)
        index += 1
    hop10 = []
    hop10_labels = []
    index = 0
    while index < len(train_ends):
        if (train_ends[index] - (WINDOW_SECONDS - 1)) % 10 == 0:
            hop10.append(train_ends[index])
            hop10_labels.append(train_labels[index])
        index += 1
    if len(hop10) == 0:
        row["skip"] = "no hop 10 train"
        return row
    n_monitor = len(hop10) // 4
    if n_monitor == 0:
        row["skip"] = "monitor empty"
        return row
    monitor_clock = hop10[len(hop10) - n_monitor]
    grad_labels = []
    monitor_labels = []
    index = 0
    while index < len(train_labels):
        if train_ends[index] < monitor_clock:
            if (train_ends[index] - (WINDOW_SECONDS - 1)) % 10 == 0:
                grad_labels.append(train_labels[index])
        else:
            monitor_labels.append(train_labels[index])
        index += 1
    n_pos_train = _count(grad_labels, LABEL_SOP)
    n_neg_train = _count(grad_labels, LABEL_INTERICTAL)
    n_pos_mon = _count(monitor_labels, LABEL_SOP)
    n_neg_mon = _count(monitor_labels, LABEL_INTERICTAL)
    row["n_positive_train"] = n_pos_train
    row["n_negative_train"] = n_neg_train
    row["n_positive_monitor"] = n_pos_mon
    row["n_negative_monitor"] = n_neg_mon
    row["n_val_windows"] = len(val_ends)
    row["n_test_windows"] = len(test_ends)
    if n_pos_train < 1:
        row["skip"] = "train 0 SOP"
        return row
    if n_neg_train < 1:
        row["skip"] = "train 0 interictal"
        return row
    if n_pos_mon < 1 or n_neg_mon < 1:
        row["skip"] = f"monitor one class pos={n_pos_mon} neg={n_neg_mon}"
        return row
    if len(test_ends) < 1:
        row["skip"] = "no test windows"
        return row
    scored = []
    _extend(scored, val_ends)
    _extend(scored, test_ends)
    try:
        _t1, _t2, test_leads, _n_merged, _n_test = test_leads_from_seizures(seizures)
        coverage = catch_coverage_from_ends(scored, test_leads, duration)
    except Exception as err:
        row["skip"] = f"catch {err}"
        return row
    for key in coverage:
        row[key] = coverage[key]
    if not coverage["catch_evaluable"]:
        row["skip"] = (
            "clock evaluable false, max lookback "
            f"{coverage['max_lookback_scores_in_hit']}"
        )
        return row
    if ram_over:
        row["skip"] = f"ram {ram_gib:.2f} GiB"
        return row
    row["clock_pass"] = True
    row["skip"] = None
    return row


def _event_paths():
    out = []
    for dirpath, _dirnames, filenames in os.walk(META_ROOT):
        index = 0
        while index < len(filenames):
            name = filenames[index]
            if name.endswith("_events.tsv"):
                out.append(os.path.join(dirpath, name))
            index += 1
    out.sort()
    return out


def _sidecar_paths(patient_id, run_id):
    base = os.path.join(META_ROOT, patient_id, "ses-01")
    stem = f"{patient_id}_ses-01_task-szMonitoring_run-{run_id}"
    return {
        "eeg": os.path.join(base, "eeg", f"{stem}_eeg.json"),
        "ecg": os.path.join(base, "ecg", f"{stem}_ecg.json"),
        "emg": os.path.join(base, "emg", f"{stem}_emg.json"),
        "mov": os.path.join(base, "mov", f"{stem}_mov.json"),
    }


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def _quality_ok(duration, drop_spans):
    ok = []
    second = 0
    while second < duration:
        hit = False
        span_i = 0
        while span_i < len(drop_spans):
            start, end = drop_spans[span_i]
            if start <= second < end:
                hit = True
                break
            span_i += 1
        ok.append(not hit)
        second += 1
    return ok


def _graph_ok(quality_ok):
    n = len(quality_ok)
    out = []
    second = 0
    while second < n:
        if second + 1 < MOVEMENT_BUFFER_SECONDS:
            out.append(False)
        elif not quality_ok[second]:
            out.append(False)
        else:
            full = True
            offset = 0
            window_start = second - MOVEMENT_BUFFER_SECONDS + 1
            while offset < MOVEMENT_BUFFER_SECONDS:
                if not quality_ok[window_start + offset]:
                    full = False
                    break
                offset += 1
            out.append(full)
        second += 1
    return out


def _count(labels, wanted):
    n = 0
    index = 0
    while index < len(labels):
        if labels[index] == wanted:
            n += 1
        index += 1
    return n


def _extend(out, values):
    index = 0
    while index < len(values):
        out.append(int(values[index]))
        index += 1


if __name__ == "__main__":
    sys.exit(main())
