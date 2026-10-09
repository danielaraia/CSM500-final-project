import json
import os

import numpy as np

from data.events import load_event_rows, seizure_spans
from evaluation.metrics import read_metrics
from packs.pack_eval import seizeit2_run_paths

SPH_SECONDS = 5 * 60
GAP_SECONDS = 60 * 60


def oracle_label(time, seizures, sop):
    if any(onset <= time < offset for onset, offset in seizures):
        return None
    if any(onset - SPH_SECONDS <= time < onset for onset, _ in seizures):
        return None
    if any(offset <= time < offset + GAP_SECONDS for _, offset in seizures):
        return None
    if any(
        onset - SPH_SECONDS - sop <= time < onset - SPH_SECONDS
        for onset, _ in seizures
    ):
        return 1
    if any(onset - GAP_SECONDS <= time < onset for onset, _ in seizures):
        return None
    return 0


def audit_packed_labels(folder, seizures, sop):
    checked = 0
    mistakes = []
    for split in ("train", "monitor", "val", "test"):
        path = os.path.join(folder, split + ".npz")
        if not os.path.isfile(path):
            continue
        with np.load(path) as packed:
            labels = packed["labels"]
            ends = packed["ends"]
        for index, (end, label) in enumerate(zip(ends, labels)):
            expected = oracle_label(float(end), seizures, sop)
            checked += 1
            if expected is None or int(label) != expected:
                mistakes.append(
                    {
                        "split": split,
                        "index": index,
                        "end": float(end),
                        "packed": int(label),
                        "expected": expected,
                    }
                )
    return {
        "n_checked": checked,
        "n_mistakes": len(mistakes),
        "mistakes": mistakes[:20],
        "all_labels_match": checked > 0 and not mistakes,
    }


def audit_one(root, patient, sop_minutes):
    folder = os.path.join(
        root,
        "experiments",
        "runs",
        "within_patient",
        patient,
        "pearson",
        f"sop{sop_minutes}",
        "seed123",
        "eval1",
    )
    metrics = read_metrics(folder)
    events = seizeit2_run_paths(patient, metrics["packed_run"])["events"]
    seizures = seizure_spans(load_event_rows(events))
    result = audit_packed_labels(folder, seizures, sop_minutes * 60)
    result.update({"patient": patient, "sop_minutes": sop_minutes})
    return result


def main():
    from tools.run_patient_grid import PATIENT_RUNS

    rows = []
    for patient in PATIENT_RUNS:
        for sop in (30, 15, 5):
            rows.append(audit_one(".", patient, sop))
    os.makedirs("results", exist_ok=True)
    with open("results/label-audit.json", "w") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
