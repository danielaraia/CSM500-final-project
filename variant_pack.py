import os
import shutil

from attention import _link_or_copy
from config import start_run
from metrics import read_metrics, write_metrics
from recipe import recipe_extra

SPLIT_NAMES = ("train", "monitor", "val", "test")
# pack side keys worth keeping in the variant folder; train side keys never
# copy over or the variant train metrics would look already written
PACK_KEYS_FROM_SOURCE = (
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
    "packed_run",
    "pack_scope",
    "train_stride_seconds",
    "eval_stride_seconds",
    "monitor_stride_seconds",
    "scored_at_train_stride",
)


def prepare_variant_folder(
    root,
    patient_id,
    source_folder,
    recipe,
    leaf,
    graph_strategy,
    sop_minutes,
    command,
):
    if not os.path.isdir(source_folder):
        raise ValueError(f"source pack folder is missing: {source_folder}")
    extra = {
        "command": command,
        "graph_strategy": graph_strategy,
        "sop_minutes": sop_minutes,
        "variant_of": source_folder,
    }
    extra.update(recipe_extra(recipe))
    folder = start_run(root, patient_id, extra=extra, leaf=leaf)
    try:
        for name in SPLIT_NAMES:
            src = os.path.join(source_folder, name + ".npz")
            dst = os.path.join(folder, name + ".npz")
            if not os.path.isfile(src):
                raise ValueError(f"source split is missing: {src}")
            _link_or_copy(src, dst)
        pack_metrics = {
            "patient_id": patient_id,
            "command": command,
            "graph_strategy": graph_strategy,
            "sop_minutes": sop_minutes,
            "variant_of": source_folder,
            "run_leaf": leaf,
        }
        pack_metrics.update(recipe_extra(recipe))
        source_metrics = read_metrics(source_folder)
        for key in PACK_KEYS_FROM_SOURCE:
            if key in source_metrics:
                pack_metrics[key] = source_metrics[key]
        write_metrics(folder, pack_metrics)
    except Exception:
        shutil.rmtree(folder)
        raise
    return folder
