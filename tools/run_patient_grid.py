import os
import shutil
import subprocess
import sys

from core.config import ALLOWED_SEEDS, run_folder_path
from packs.pack_eval import seizeit2_run_paths
from packs.pack_p6 import download_run_files
from tools.label_audit import audit_one
from train.train_sop import NEW_SOP_MINUTES, PATIENTS, TRAIN_GRAPHS

PATIENT_RUNS = {
    "sub-004": "01",
    "sub-015": "04",
    "sub-039": "06",
    "sub-051": "01",
    "sub-052": "07",
    "sub-073": "50",
    "sub-113": "04",
}


def run(*parts):
    subprocess.run([sys.executable, "-m", *map(str, parts)], check=True)


def pack_patient(patient):
    run_id = PATIENT_RUNS[patient]
    missing = []
    valid = set()
    for sop in NEW_SOP_MINUTES:
        for graph in ("pearson", "mi"):
            folder = run_folder_path(".", patient, "eval1", graph, sop, 123)
            metrics = os.path.join(folder, "metrics.json")
            if os.path.isfile(metrics):
                valid.add((sop, graph))
            else:
                shutil.rmtree(folder, ignore_errors=True)
                missing.append((sop, graph))
    if missing:
        paths = seizeit2_run_paths(patient, run_id)
        download_run_files(patient, run_id, paths)
        for sop, graph in missing:
            try:
                run("packs.pack_sop", patient, run_id, graph, sop)
                valid.add((sop, graph))
            except subprocess.CalledProcessError:
                folder = run_folder_path(
                    ".", patient, "eval1", graph, sop, 123
                )
                shutil.rmtree(folder, ignore_errors=True)
                print("skipped pack", patient, graph, sop)
    for sop in NEW_SOP_MINUTES:
        if (sop, "pearson") in valid:
            result = audit_one(".", patient, sop)
            if not result["all_labels_match"]:
                raise ValueError(f"label audit failed: {patient} sop{sop}")
    return valid


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PATIENTS:
        raise ValueError("usage: python -m tools.run_patient_grid patient")
    patient = sys.argv[1]
    valid = pack_patient(patient)
    for sop in NEW_SOP_MINUTES:
        graphs = []
        if (sop, "pearson") in valid:
            graphs += ["pearson", "attention", "identity"]
        if (sop, "mi") in valid:
            graphs += ["mi"]
        for graph in graphs:
            for seed in ALLOWED_SEEDS:
                run("train.train_sop", patient, graph, sop, seed)
                run("evaluation.score_sop", patient, graph, sop, seed)
        if (sop, "pearson") in valid:
            run("train.train_logistic", patient, sop)


if __name__ == "__main__":
    main()
