import json
import time

from metrics import write_metrics
from run import write_split_run


def main():
    base = "data/seizeit2/sub-004/ses-01"
    extra = {
        "command": "python pack_eval.py",
        "pack_scope": "full recording",
        "negative_control": "none",
    }
    clock = time.perf_counter()
    folder, counts = write_split_run(
        ".",
        "sub-004",
        f"{base}/eeg/sub-004_ses-01_task-szMonitoring_run-01_eeg.edf",
        f"{base}/ecg/sub-004_ses-01_task-szMonitoring_run-01_ecg.edf",
        f"{base}/emg/sub-004_ses-01_task-szMonitoring_run-01_emg.edf",
        f"{base}/mov/sub-004_ses-01_task-szMonitoring_run-01_mov.edf",
        f"{base}/eeg/sub-004_ses-01_task-szMonitoring_run-01_events.tsv",
        extra=extra,
        leaf="eval1",
    )
    metrics = {
        "patient_id": "sub-004",
        "command": "python pack_eval.py",
        "pack_scope": "full recording",
        "pack_seconds": float(time.perf_counter() - clock),
        "negative_control": "none",
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
