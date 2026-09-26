"""Rebuild the labels from raw events and check them against the packer metrics."""

import glob
import json
import math
import os

from data.events import load_event_rows, seizure_spans
from data.labels import (
    LABEL_SOP,
    SPH_SECONDS,
    label_second,
    seizures_with_full_sop,
    split_cut_times,
)
from packs.pack_eval import seizeit2_run_paths

# the 8th handled patient (sub-022) has no local events tsv, so it is audited
# on the pod by the replication run, not here
HANDLED = (
    "sub-004",
    "sub-015",
    "sub-039",
    "sub-051",
    "sub-052",
    "sub-073",
    "sub-113",
)
SOP_MINUTES = (30, 15, 10, 5)
# labels are graph independent, so any graph's eval1 metrics.json will do; pick
# the first that exists in this order
GRAPHS = ("pearson", "mi", "attention")
T_TOLERANCE = 1.0


def find_metrics(root, patient_id, sop_minutes):
    for graph in GRAPHS:
        path = os.path.join(
            root,
            "experiments",
            "runs",
            "within_patient",
            patient_id,
            graph,
            f"sop{sop_minutes}",
            "seed123",
            "eval1",
            "metrics.json",
        )
        if os.path.isfile(path):
            return path, graph
    return None, None


def sop_second_counts_by_region(all_seizures, sop_seconds, t1, t2):
    # sop seconds only live in each seizure's [onset-sph-sop, onset-sph) window,
    # so scan just those bounded intervals rather than the whole recording; a
    # second still resolves through the full label_second precedence
    seen = set()
    counts = {"train": 0, "val": 0, "test": 0}
    for onset, _offset in all_seizures:
        start = int(math.floor(onset - SPH_SECONDS - sop_seconds))
        stop = int(math.ceil(onset - SPH_SECONDS))
        for second in range(max(start, 0), stop):
            if second in seen:
                continue
            seen.add(second)
            if label_second(second, all_seizures, sop_seconds) != LABEL_SOP:
                continue
            if second < t1:
                counts["train"] += 1
            elif second < t2:
                counts["val"] += 1
            else:
                counts["test"] += 1
    return counts


def audit_one(root, patient_id, sop_minutes):
    metrics_path, graph = find_metrics(root, patient_id, sop_minutes)
    if metrics_path is None:
        return None
    with open(metrics_path) as handle:
        metrics = json.load(handle)
    run_id = metrics.get("packed_run")
    if run_id is not None:
        events_path = os.path.join(
            root, seizeit2_run_paths(patient_id, run_id)["events"]
        )
    else:
        # early packs (e.g. sub-004) did not record packed_run; each handled
        # patient has exactly one local events tsv, so fall back to it
        matches = sorted(
            glob.glob(
                os.path.join(
                    root, "data", "seizeit2", patient_id, "ses-01", "eeg", "*_events.tsv"
                )
            )
        )
        events_path = matches[0] if matches else ""
        run_id = "local-fallback"
    if not events_path or not os.path.isfile(events_path):
        return {
            "patient_id": patient_id,
            "sop_minutes": sop_minutes,
            "status": "no_events_tsv",
            "events_path": events_path,
        }
    sop_seconds = sop_minutes * 60
    rows = load_event_rows(events_path)
    all_seizures = seizure_spans(rows)
    usable = seizures_with_full_sop(all_seizures, sop_seconds)
    t1, t2, n_train_seizures, n_test_seizures = split_cut_times(usable, sop_seconds)
    sop_seconds_region = sop_second_counts_by_region(all_seizures, sop_seconds, t1, t2)

    # early packs predate these metric keys, so only check what got recorded
    checks = {}
    if "t1" in metrics:
        checks["t1"] = abs(t1 - metrics["t1"]) < T_TOLERANCE
    if "t2" in metrics:
        checks["t2"] = abs(t2 - metrics["t2"]) < T_TOLERANCE
    if "n_train_seizures" in metrics:
        checks["n_train_seizures"] = n_train_seizures == metrics["n_train_seizures"]
    if "n_test_seizures" in metrics:
        checks["n_test_seizures"] = n_test_seizures == metrics["n_test_seizures"]

    positive_findings = {}
    for region, pos_key in (
        ("train", "n_positive_train"),
        ("val", "n_positive_val"),
        ("test", "n_positive_test"),
    ):
        recomputed = sop_seconds_region[region]
        if pos_key not in metrics:
            positive_findings[region] = {
                "recomputed_sop_seconds": recomputed,
                "packed_positive_windows": None,
                "verdict": "metrics_key_absent",
            }
            continue
        packed = metrics[pos_key]
        if recomputed == 0 and packed == 0:
            verdict = "structural_no_sop_seconds"
        elif recomputed > 0 and packed > 0:
            verdict = "positives_present"
        elif recomputed > 0 and packed == 0:
            verdict = "qc_removed_all_positives"
        else:
            verdict = "IMPOSSIBLE_positives_without_sop_seconds"
        positive_findings[region] = {
            "recomputed_sop_seconds": recomputed,
            "packed_positive_windows": packed,
            "verdict": verdict,
        }

    return {
        "patient_id": patient_id,
        "sop_minutes": sop_minutes,
        "status": "audited",
        "graph_used": graph,
        "packed_run": run_id,
        "n_seizures_total": len(all_seizures),
        "n_seizures_usable": len(usable),
        "seizure_spans": [[round(o, 2), round(f, 2)] for o, f in all_seizures],
        "recomputed": {
            "t1": t1,
            "t2": t2,
            "n_train_seizures": n_train_seizures,
            "n_test_seizures": n_test_seizures,
        },
        "packed": {key: metrics.get(key) for key in (
            "t1",
            "t2",
            "n_train_seizures",
            "n_test_seizures",
            "n_positive_train",
            "n_negative_train",
            "n_positive_val",
            "n_positive_test",
            "n_negative_test",
        )},
        "checks": checks,
        "positive_findings": positive_findings,
        "all_checks_pass": all(checks.values())
        and all(
            f["verdict"] != "IMPOSSIBLE_positives_without_sop_seconds"
            for f in positive_findings.values()
        ),
    }


def run_audit(root):
    results = []
    for patient_id in HANDLED:
        for sop_minutes in SOP_MINUTES:
            one = audit_one(root, patient_id, sop_minutes)
            if one is not None:
                results.append(one)
    return results


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    results = run_audit(root)
    out_path = os.path.join(root, "label-audit.json")
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)

    audited = [r for r in results if r["status"] == "audited"]
    n_pass = sum(1 for r in audited if r["all_checks_pass"])
    print(f"audited {len(audited)} patient-sop combos, {n_pass} fully consistent")
    header = (
        f"{'patient':9} {'sop':4} {'t1':>3} {'t2':>3} {'sz':>4} "
        f"{'train':>18} {'val':>18} {'test':>22}"
    )
    print(header)
    for r in audited:
        c = r["checks"]
        tick = lambda ok: "ok" if ok else "FAIL"
        pf = r["positive_findings"]
        short = {
            "structural_no_sop_seconds": "none(structural)",
            "positives_present": "present",
            "qc_removed_all_positives": "QC-killed",
            "IMPOSSIBLE_positives_without_sop_seconds": "IMPOSSIBLE",
            "metrics_key_absent": "n/a(old-pack)",
        }
        sz_ok = c.get("n_train_seizures", True) and c.get("n_test_seizures", True)
        print(
            f"{r['patient_id']:9} {r['sop_minutes']:<4} "
            f"{tick(c.get('t1', True)):>3} {tick(c.get('t2', True)):>3} "
            f"{tick(sz_ok):>4} "
            f"{short[pf['train']['verdict']]:>18} "
            f"{short[pf['val']['verdict']]:>18} "
            f"{short[pf['test']['verdict']]:>22}"
        )
    skipped = [r for r in results if r["status"] != "audited"]
    for r in skipped:
        print(f"  skipped {r['patient_id']} sop{r['sop_minutes']}: {r['status']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
