import json
import os

import numpy as np
import torch

from alarms import load_window_scores
from constant_check import (
    checkpoint_health,
    class_mix,
    ends_time_ordered,
    graph_spread,
    labels_match_clock,
    state_dicts_max_abs_diff,
    tiny_batch_learning,
)
from events import load_event_rows, seizure_spans
from labels import seizures_with_full_sop
from metrics import read_metrics
from pos_weight import weight_from_class_numbers
from score_alarms import KNOWN_EVENTS
from windows import TRAIN_STRIDE_SECONDS, end_is_on_stride


PACKED_FOLDERS = (
    "experiments/runs/within_patient/sub-051/pearson/sop30/seed123/eval1",
    "experiments/runs/within_patient/sub-052/pearson/sop30/seed123/eval1",
)

SPLIT_NAMES = ("train", "monitor", "val", "test")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    summary = {"command": "python report_constant.py", "patients": []}
    states = []
    index = 0
    while index < len(PACKED_FOLDERS):
        folder = os.path.join(here, PACKED_FOLDERS[index])
        row, state = check_folder(folder, here)
        summary["patients"].append(row)
        states.append(state)
        print("folder", folder)
        print(json.dumps(row, indent=2))
        index += 1
    ckpt_diff = state_dicts_max_abs_diff(states[0], states[1])
    summary["checkpoint_diff"] = ckpt_diff
    summary["verdict"] = _verdict(summary["patients"], ckpt_diff)
    out = os.path.join(here, "p2-constant-check-summary.json")
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"verdict": summary["verdict"], "checkpoint_diff": ckpt_diff}, indent=2))


def check_folder(folder, repo_root):
    metrics = read_metrics(folder)
    patient_id = metrics["patient_id"]
    events_rel = KNOWN_EVENTS.get(patient_id)
    if events_rel is None:
        raise ValueError(f"no events path for {patient_id}")
    seizures = seizures_with_full_sop(
        seizure_spans(load_event_rows(os.path.join(repo_root, events_rel)))
    )
    splits = {}
    prev_last = None
    name_index = 0
    while name_index < len(SPLIT_NAMES):
        name = SPLIT_NAMES[name_index]
        pack = np.load(os.path.join(folder, name + ".npz"), mmap_mode="r")
        labels = np.asarray(pack["labels"])
        ends = np.asarray(pack["ends"])
        mix = class_mix(labels)
        scores_ends, logits = load_window_scores(os.path.join(folder, name + "_scores.npz"))
        n_hop10 = 0
        i = 0
        while i < ends.shape[0]:
            if end_is_on_stride(int(ends[i]), TRAIN_STRIDE_SECONDS):
                n_hop10 += 1
            i += 1
        after_prev = True
        if prev_last is not None and ends.shape[0] > 0:
            after_prev = int(ends[0]) > int(prev_last)
        if ends.shape[0] > 0:
            prev_last = int(ends[-1])
        splits[name] = {
            "n_windows": int(ends.shape[0]),
            "n_sop": mix["n_sop"],
            "n_interictal": mix["n_interictal"],
            "mixed": mix["mixed"],
            "ends_time_ordered": bool(ends_time_ordered(ends)),
            "n_clock_mismatch": int(labels_match_clock(ends, labels, seizures)),
            "n_score_end_mismatch": int(
                np.sum(ends.astype(int) != np.asarray(scores_ends).astype(int))
            ),
            "after_prev_split": bool(after_prev),
            "n_hop10": int(n_hop10),
            "graph": graph_spread(pack["adjacency"]),
            "logit_mean": float(np.mean(logits)),
            "logit_std": float(np.std(logits)),
        }
        name_index += 1
    train = np.load(os.path.join(folder, "train.npz"), mmap_mode="r")
    labels = np.asarray(train["labels"])
    sop_i = np.where(labels == 1)[0][:4]
    int_i = np.where(labels == 0)[0][:4]
    idx = np.concatenate([int_i, sop_i])
    pos_weight, _n_neg, _n_pos = weight_from_class_numbers(labels)
    learn = tiny_batch_learning(
        np.asarray(train["features"][idx]),
        np.asarray(train["adjacency"][idx]),
        labels[idx],
        pos_weight,
        n_steps=5,
    )
    ckpt_path = os.path.join(folder, "checkpoints", "best.pth.tar")
    ckpt = checkpoint_health(ckpt_path)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    row = {
        "patient_id": patient_id,
        "splits": splits,
        "tiny_batch": learn,
        "checkpoint": ckpt,
        "pos_weight": float(pos_weight),
    }
    return row, blob["model"]


def _verdict(patients, ckpt_diff):
    labels_ok = True
    graph_ok = True
    learn_ok = True
    p = 0
    while p < len(patients):
        row = patients[p]
        s = 0
        while s < len(SPLIT_NAMES):
            name = SPLIT_NAMES[s]
            split = row["splits"][name]
            if name != "val" and not split["mixed"]:
                labels_ok = False
            if name == "val" and split["n_sop"] != 0:
                labels_ok = False
            if not split["ends_time_ordered"]:
                labels_ok = False
            if split["n_clock_mismatch"] != 0:
                labels_ok = False
            if split["n_score_end_mismatch"] != 0:
                labels_ok = False
            if not split["after_prev_split"]:
                labels_ok = False
            if split["graph"]["dead"]:
                graph_ok = False
            s += 1
        tb = row["tiny_batch"]
        ck = row["checkpoint"]
        if not tb["grad_nonzero"] or not tb["weight_moved"]:
            learn_ok = False
        if not ck["finite"] or ck["n_all_zero_tensors"] != 0:
            learn_ok = False
        if ck["optimizer_max_exp_avg"] <= 0.0:
            learn_ok = False
        p += 1
    if ckpt_diff["max_abs_diff"] <= 0.0:
        learn_ok = False
    if labels_ok and graph_ok and learn_ok:
        return "honest constant classifier"
    return "bug"


if __name__ == "__main__":
    main()
