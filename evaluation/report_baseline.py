import json
import os

from evaluation.score_baseline import run_folder


PACKED_FOLDERS = (
    "experiments/runs/within_patient/sub-051/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
)

# most SOP windows must sit above every interictal window
SEPARATE_FRAC = 0.5


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary = {
        "command": "python report_baseline.py",
        "gpu": False,
        "patients": [],
    }
    index = 0
    while index < len(PACKED_FOLDERS):
        folder = os.path.join(here, PACKED_FOLDERS[index])
        row = run_folder(folder, here)
        summary["patients"].append(row)
        print("folder", folder)
        print(json.dumps(row, indent=2))
        index += 1
    summary["verdict"] = plain_verdict(summary["patients"])
    out = os.path.join(here, "results/p2-simple-baseline-summary.json")
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"verdict": summary["verdict"]}, indent=2))
    _print_table(summary["patients"])


def plain_verdict(patients):
    index = 0
    while index < len(patients):
        base = patients[index]["baseline"]
        if base.get("study_criterion_met"):
            return "recipe hard"
        overlap = base.get("logit_overlap") or {}
        test = overlap.get("test") or {}
        frac = test.get("frac_sop_at_or_under_interictal_max")
        if frac is not None and frac < SEPARATE_FRAC:
            return "recipe hard"
        index += 1
    return "task hard"


def _print_table(patients):
    print(
        "person model sens fa_per_hour test_overlap train_overlap "
        "threshold_set caught missed unevaluable"
    )
    index = 0
    while index < len(patients):
        row = patients[index]
        _print_model_line(row["patient_id"], "evobrain", row["evobrain"])
        _print_model_line(row["patient_id"], "logistic", row["baseline"])
        index += 1


def _print_model_line(patient_id, name, block):
    test_frac = None
    train_frac = None
    overlap = block.get("logit_overlap")
    if isinstance(overlap, dict):
        if "test" in overlap:
            test_frac = overlap["test"].get("frac_sop_at_or_under_interictal_max")
        if "train" in overlap:
            train_frac = overlap["train"].get("frac_sop_at_or_under_interictal_max")
    print(
        patient_id,
        name,
        block.get("event_sensitivity"),
        block.get("false_alarms_per_hour"),
        test_frac,
        train_frac,
        block.get("threshold_set"),
        block.get("n_seizures_caught"),
        block.get("n_seizures_missed"),
        block.get("n_seizures_unevaluable"),
    )


if __name__ == "__main__":
    main()
