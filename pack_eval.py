import json
import os
import time

from metrics import write_metrics
from run import write_split_run

PATIENT_ID = "sub-007"
RUN_ID = "03"
LEAF = "eval1"


def seizeit2_run_paths(patient_id, run_id):
    base = os.path.join("data", "seizeit2", patient_id, "ses-01")
    stem = f"{patient_id}_ses-01_task-szMonitoring_run-{run_id}"
    return {
        "eeg": os.path.join(base, "eeg", f"{stem}_eeg.edf"),
        "ecg": os.path.join(base, "ecg", f"{stem}_ecg.edf"),
        "emg": os.path.join(base, "emg", f"{stem}_emg.edf"),
        "mov": os.path.join(base, "mov", f"{stem}_mov.edf"),
        "events": os.path.join(base, "eeg", f"{stem}_events.tsv"),
    }


def main():
    paths = seizeit2_run_paths(PATIENT_ID, RUN_ID)
    extra = {
        "command": "python pack_eval.py",
        "pack_scope": "seizure bearing run only",
        "negative_control": "none",
        "packed_run": RUN_ID,
    }
    clock = time.perf_counter()
    folder, counts = write_split_run(
        ".",
        PATIENT_ID,
        paths["eeg"],
        paths["ecg"],
        paths["emg"],
        paths["mov"],
        paths["events"],
        extra=extra,
        leaf=LEAF,
    )
    metrics = {
        "patient_id": PATIENT_ID,
        "command": "python pack_eval.py",
        "pack_scope": "seizure bearing run only",
        "pack_seconds": float(time.perf_counter() - clock),
        "negative_control": "none",
        "packed_run": RUN_ID,
    }
    for key in counts:
        if key == "means" or key == "stds":
            continue
        metrics[key] = counts[key]
    write_metrics(folder, metrics)
    print("folder", folder)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
