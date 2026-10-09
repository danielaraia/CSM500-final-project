import os
import sys
import time

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from core.config import run_folder_path
from evaluation.alarms import save_window_scores
from evaluation.metrics import read_metrics, replace_metrics_keys
from evaluation.score_sop import score_one_folder
from packs.pack import load_split
from packs.seed_pack import prepare_seed_folder
from train.train_sop import NEW_SOP_MINUTES, PATIENTS


def parse_args(argv):
    if len(argv) != 3:
        raise ValueError(
            "usage: python -m train.train_logistic patient sop"
        )
    patient, sop = argv[1], int(argv[2])
    if patient not in PATIENTS:
        raise ValueError(f"unknown patient: {patient}")
    if sop not in NEW_SOP_MINUTES:
        raise ValueError(f"unknown sop: {sop}")
    return patient, sop


def summary_features(features):
    width = features.shape[2] * features.shape[3]
    return features.mean(axis=1).reshape(len(features), width)


def main():
    patient, sop = parse_args(sys.argv)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = run_folder_path(root, patient, "eval1", "pearson", sop, 123)
    folder = run_folder_path(root, patient, "eval1", "logistic", sop, 123)
    if not os.path.isdir(folder):
        folder = prepare_seed_folder(
            root, patient, source, "logistic", sop, 123
        )
    if os.path.isfile(os.path.join(folder, "test_scores.npz")):
        score_one_folder(root, patient, "logistic", sop, 123)
        print("already trained", folder)
        return

    splits = {}
    for name in ("train", "monitor", "val", "test"):
        features, _graphs, labels, ends = load_split(
            os.path.join(folder, name + ".npz")
        )
        splits[name] = (summary_features(features), labels, ends)

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=123
    )
    started = time.perf_counter()
    model.fit(splits["train"][0], splits["train"][1])
    train_seconds = time.perf_counter() - started

    for name, (features, _labels, ends) in splits.items():
        scores = (
            model.decision_function(features)
            if len(features)
            else np.asarray([])
        )
        save_window_scores(
            os.path.join(folder, name + "_scores.npz"), ends, scores
        )
    np.savez_compressed(
        os.path.join(folder, "logistic_model.npz"),
        coef=model.coef_,
        intercept=model.intercept_,
    )
    replace_metrics_keys(
        folder,
        {
            "train_model": "logistic regression",
            "graph_strategy": "logistic",
            "seed": 123,
            "train_seconds": train_seconds,
            "parameter_count": int(model.coef_.size + model.intercept_.size),
            "logistic_iterations": int(model.n_iter_[0]),
            "sklearn_version": sklearn.__version__,
        },
    )
    score_one_folder(root, patient, "logistic", sop, 123)
    print("trained", folder)


if __name__ == "__main__":
    main()
