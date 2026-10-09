import json
import math
import os
import sys

import numpy as np

from core.config import ALLOWED_SEEDS, run_folder_path
from evaluation.metrics import read_metrics
from train.train_sop import NEW_SOP_MINUTES, PATIENTS, TRAIN_GRAPHS

GRAPHS = TRAIN_GRAPHS + ("logistic",)
METRICS = ("event_sensitivity", "false_alarms_per_hour")


def seeds_for(graph):
    return (123,) if graph == "logistic" else ALLOWED_SEEDS


def collect_rows(root):
    rows = []
    for patient in PATIENTS:
        for graph in GRAPHS:
            for sop in NEW_SOP_MINUTES:
                for seed in seeds_for(graph):
                    folder = run_folder_path(
                        root, patient, "eval1", graph, sop, seed
                    )
                    if not os.path.isfile(
                        os.path.join(folder, "metrics.json")
                    ):
                        continue
                    data = read_metrics(folder)
                    rows.append(
                        {
                            "patient": patient,
                            "graph": graph,
                            "sop": sop,
                            "seed": seed,
                            "event_sensitivity": data["event_sensitivity"],
                            "false_alarms_per_hour": data[
                                "false_alarms_per_hour"
                            ],
                            "caught": data["n_seizures_caught"],
                            "seizures": data["n_seizures_evaluable"],
                        }
                    )
    return rows


def expected_rows(root):
    total = 0
    for patient in PATIENTS:
        for sop in NEW_SOP_MINUTES:
            pearson = run_folder_path(
                root, patient, "eval1", "pearson", sop, 123
            )
            mi = run_folder_path(root, patient, "eval1", "mi", sop, 123)
            if os.path.isfile(os.path.join(pearson, "metrics.json")):
                total += 10
            if os.path.isfile(os.path.join(mi, "metrics.json")):
                total += 3
    return total


def patient_means(rows):
    groups = {}
    for row in rows:
        key = row["patient"], row["graph"], row["sop"]
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        item = {"patient": key[0], "graph": key[1], "sop": key[2]}
        for metric in METRICS:
            values = [
                row[metric] for row in group if row[metric] is not None
            ]
            item[metric] = None if not values else float(np.mean(values))
            item[metric + "_seed_std"] = (
                None
                if len(values) < 2
                else float(np.std(values, ddof=1))
            )
        result.append(item)
    return result


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(123)
    picks = rng.choice(values, size=(10000, len(values)), replace=True)
    means = picks.mean(axis=1)
    return {
        "n_patients": len(values),
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def summaries(patient_rows):
    result = []
    for graph in GRAPHS:
        for sop in NEW_SOP_MINUTES:
            group = [
                row
                for row in patient_rows
                if row["graph"] == graph and row["sop"] == sop
            ]
            if not group:
                continue
            item = {"graph": graph, "sop": sop}
            for metric in METRICS:
                values = [
                    row[metric] for row in group if row[metric] is not None
                ]
                if values:
                    item[metric] = interval(values)
            result.append(item)
    return result


def comparisons(patient_rows):
    lookup = {
        (row["patient"], row["graph"], row["sop"]): row
        for row in patient_rows
    }
    result = []
    for graph in ("pearson", "mi", "attention"):
        for control in ("identity", "logistic"):
            for sop in NEW_SOP_MINUTES:
                item = {"graph": graph, "control": control, "sop": sop}
                for metric in METRICS:
                    differences = [
                        lookup[(patient, graph, sop)][metric]
                        - lookup[(patient, control, sop)][metric]
                        for patient in PATIENTS
                        if (patient, graph, sop) in lookup
                        and (patient, control, sop) in lookup
                        and lookup[(patient, graph, sop)][metric] is not None
                        and lookup[(patient, control, sop)][metric] is not None
                    ]
                    if not differences:
                        continue
                    item[metric] = interval(differences)
                if len(item) > 3:
                    result.append(item)
    return result


def chance_rows(rows):
    result = []
    for row in rows:
        p = 1.0 - math.exp(-0.15 * row["sop"] / 60.0)
        n, caught = row["seizures"], row["caught"]
        tail = sum(
            math.comb(n, k) * p**k * (1 - p) ** (n - k)
            for k in range(caught, n + 1)
        )
        result.append(
            {
                "patient": row["patient"],
                "graph": row["graph"],
                "sop": row["sop"],
                "seed": row["seed"],
                "chance_hit_probability": p,
                "chance_p_at_least_observed": tail,
            }
        )
    return result


def main():
    if len(sys.argv) > 2:
        raise ValueError("usage: python -m evaluation.experiment_analysis [runs]")
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as handle:
            rows = json.load(handle)
    else:
        rows = collect_rows(".")
        expected = expected_rows(".")
        if len(rows) != expected:
            raise ValueError(f"expected {expected} runs, got {len(rows)}")
    patients = patient_means(rows)
    result = {
        "patient_means": patients,
        "summaries": summaries(patients),
        "comparisons": comparisons(patients),
        "chance": chance_rows(rows),
        "note": "Seeds are averaged first. Bootstrap intervals use patients.",
    }
    os.makedirs("results", exist_ok=True)
    with open("results/experiment-analysis.json", "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"analysed {len(rows)} runs")


if __name__ == "__main__":
    main()
