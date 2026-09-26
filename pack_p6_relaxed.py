"""Pack the relaxed hours cohort at SOP 15, 10, 5. One ledger row per attempt."""

import json
import os
import sys

from core.config import run_folder_path
from metrics import read_metrics
from pack_eval import seizeit2_run_paths
from pack_p6 import (
    GRAPH_STRATEGY,
    append_ledger,
    download_run_files,
    ledger_row,
    load_cohort,
    pack_one,
    r2_pack_path,
    upload_pack,
)
from screen_p6 import RAM_GIB_SKIP

COHORT_NAME = "p6-screening-cohort-relaxed.json"
LEDGER_NAME = "screening-ledger.jsonl"
PACK_SOP_MINUTES = (15, 10, 5)
PHASE = "p6-pack-relaxed"


def relaxed_ledger_row(entry, sop_minutes, status, reason, r2_path, metrics=None):
    row = ledger_row(
        entry["patient_id"],
        entry["run_id"],
        sop_minutes,
        status,
        reason,
        r2_path,
        metrics,
    )
    row["phase"] = PHASE
    row["fa_resolution"] = entry["fa_resolution"]
    row["test_interictal_hours"] = entry["test_interictal_hours"]
    return row


def sop_list(only_sop=None):
    if only_sop is None:
        return list(PACK_SOP_MINUTES)
    if only_sop not in PACK_SOP_MINUTES:
        raise ValueError(
            f"--sop has to be one of {PACK_SOP_MINUTES}, we got {only_sop}"
        )
    return [only_sop]


def pack_patient(entry, allow_big=False, only_sop=None, ledger_path=LEDGER_NAME):
    patient_id = entry["patient_id"]
    run_id = entry["run_id"]
    sop_minutes_list = sop_list(only_sop)
    if entry.get("needs_pod") and not allow_big:
        for sop_minutes in sop_minutes_list:
            row = relaxed_ledger_row(
                entry,
                sop_minutes,
                "deferred_big_ram",
                f"ram estimate {entry.get('ram_gib')} GiB >= {RAM_GIB_SKIP} GiB, "
                "pack on a pod with --allow-big",
                None,
            )
            append_ledger(row, ledger_path)
            print(json.dumps(row), flush=True)
        return
    paths = seizeit2_run_paths(patient_id, run_id)
    downloaded = download_run_files(patient_id, run_id, paths)
    if downloaded:
        print(f"downloaded {patient_id} run {run_id}: {downloaded}", flush=True)
    for sop_minutes in sop_minutes_list:
        folder = run_folder_path(
            ".", patient_id, "eval1",
            graph_strategy=GRAPH_STRATEGY, sop_minutes=sop_minutes,
        )
        r2_path = r2_pack_path(patient_id, sop_minutes)
        if os.path.isfile(os.path.join(folder, "metrics.json")):
            print(f"pack exists, uploading: {folder}", flush=True)
            status, reason = "packed", None
        else:
            status, reason, folder = pack_one(patient_id, run_id, sop_minutes)
        metrics = None
        if status == "packed":
            upload_pack(folder, r2_path)
            metrics = read_metrics(folder)
        else:
            r2_path = None
        row = relaxed_ledger_row(entry, sop_minutes, status, reason, r2_path, metrics)
        append_ledger(row, ledger_path)
        print(json.dumps(row), flush=True)


def parse_args(argv):
    allow_big = "--allow-big" in argv
    only_sop = None
    rest = []
    args = list(argv[1:])
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--allow-big":
            index += 1
            continue
        if arg == "--sop":
            only_sop = int(args[index + 1])
            index += 2
            continue
        rest.append(arg)
        index += 1
    if len(rest) == 0:
        only = None
    elif len(rest) == 2:
        only = (rest[0], rest[1])
    else:
        raise ValueError(
            "usage: python pack_p6_relaxed.py [patient_id run_id] "
            "[--sop N] [--allow-big]"
        )
    sop_list(only_sop)  # validates the --sop choice
    return only, only_sop, allow_big


def main():
    only, only_sop, allow_big = parse_args(sys.argv)
    for entry in load_cohort(COHORT_NAME):
        if only is not None and (entry["patient_id"], entry["run_id"]) != only:
            continue
        pack_patient(entry, allow_big=allow_big, only_sop=only_sop)


if __name__ == "__main__":
    main()
