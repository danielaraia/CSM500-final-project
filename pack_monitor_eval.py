import json
import os
import time

from metrics import read_metrics
from pack import load_split
from run import rewrite_monitor_eval
from windows import end_is_on_stride, TRAIN_STRIDE_SECONDS


PACKED = (
    (
        "experiments/runs/within_patient/sub-004/pearson/sop30/seed123/eval1",
        "01",
    ),
    (
        "experiments/runs/within_patient/sub-051/pearson/sop30/seed123/eval1",
        "01",
    ),
    (
        "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
        "07",
    ),
    (
        "experiments/runs/within_patient/sub-113/pearson/sop30/seed123/eval1",
        "04",
    ),
)


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


def val_is_hop_one(folder):
    _f, _g, _y, ends = load_split(os.path.join(folder, "val.npz"))
    if ends is None or len(ends) == 0:
        return True
    off = 0
    index = 0
    while index < len(ends):
        if not end_is_on_stride(int(ends[index]), TRAIN_STRIDE_SECONDS):
            off += 1
        index += 1
    return off > 0 or len(ends) == 1


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    summary = {"command": "python pack_monitor_eval.py", "patients": []}
    for rel, run_id in PACKED:
        folder = os.path.join(here, rel)
        metrics = read_metrics(folder)
        patient_id = metrics["patient_id"]
        if not val_is_hop_one(folder):
            raise ValueError(f"val is still hop 10, {folder}")
        paths = seizeit2_run_paths(patient_id, run_id)
        clock = time.perf_counter()
        extra = rewrite_monitor_eval(
            folder,
            os.path.join(here, paths["eeg"]),
            os.path.join(here, paths["ecg"]),
            os.path.join(here, paths["emg"]),
            os.path.join(here, paths["mov"]),
            os.path.join(here, paths["events"]),
        )
        row = {
            "patient_id": patient_id,
            "n_monitor_windows": extra["n_monitor_windows"],
            "n_positive_monitor": extra["n_positive_monitor"],
            "n_negative_monitor": extra["n_negative_monitor"],
            "monitor_stride_seconds": extra["monitor_stride_seconds"],
            "monitor_clock": extra["monitor_clock"],
            "pack_seconds": float(time.perf_counter() - clock),
        }
        summary["patients"].append(row)
        print("folder", folder)
        print(json.dumps(row, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
