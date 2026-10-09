import os
import shutil

from core.config import start_run
from evaluation.metrics import read_metrics, write_metrics

PACK_KEYS = (
    "pack_scope",
    "pack_seconds",
    "packed_run",
    "t1",
    "t2",
    "n_train_seizures",
    "n_test_seizures",
    "n_train_windows",
    "n_monitor_windows",
    "n_validation_windows",
    "n_test_windows",
    "n_positive_train",
    "n_negative_train",
    "n_positive_monitor",
    "n_negative_monitor",
    "n_positive_val",
    "n_negative_val",
    "n_positive_test",
    "n_negative_test",
    "train_stride_seconds",
    "monitor_stride_seconds",
    "eval_stride_seconds",
    "scored_at_train_stride",
    "catch_evaluable",
    "n_test_leads",
    "n_test_leads_evaluable",
    "n_test_leads_unevaluable",
    "max_lookback_scores_in_hit",
    "n_scorable_hit_seconds",
    "alarm_lookback_seconds",
    "alarm_vote_need",
)


def prepare_seed_folder(root, patient, source, graph, sop, seed):
    folder = start_run(
        root,
        patient,
        {"graph_strategy": graph, "sop_minutes": sop, "seed": seed},
        "eval1",
    )
    try:
        for name in ("train", "monitor", "val", "test"):
            old = os.path.join(source, name + ".npz")
            new = os.path.join(folder, name + ".npz")
            if not os.path.isfile(old):
                raise ValueError(f"pack is missing: {old}")
            os.symlink(os.path.abspath(old), new)
        source_metrics = read_metrics(source)
        metrics = {
            key: value
            for key, value in source_metrics.items()
            if key in PACK_KEYS or key.startswith("n_steps_")
        }
        metrics.update(
            {
                "patient_id": patient,
                "graph_strategy": graph,
                "sop_minutes": sop,
                "seed": seed,
                "feature_pack_source": source,
            }
        )
        write_metrics(folder, metrics)
    except Exception:
        shutil.rmtree(folder)
        raise
    return folder
