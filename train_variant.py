import json
import os
import sys
import time
import traceback

import torch

from graphs.attention import load_attention_evobrain
from data.augment import make_augment_fn
from core.config import ALLOWED_SOP_MINUTES, LOCKED
from epoch_trace import install_logged_train_loop
from model.evobrain_load import load_evobrain
from model.focal_loss import make_loss_fn
from metrics import read_metrics, replace_metrics_keys
from model.pretrain import pretrain_model
from core.recipe import normalize_recipe, recipe_extra, recipe_leaf
from run import train_packed_eval_run
from data.sensor_mask import make_sensor_mask_fn
from time_cuda import make_batch
from train_sop import _epoch_keys, _ready_folder, _save_train_scores
from model.train_step import make_optimizer, one_train_step
from packs.variant_pack import prepare_variant_folder

FLAG_NAMES = {
    "--loss": "loss",
    "--augment": "augmentation",
    "--pretrain": "pretrain",
    "--sensors": "sensors",
}
USAGE = (
    "usage: python train_variant.py patient_id graph_strategy sop_minutes "
    "[--loss weighted_bce|focal] [--augment none|jitter_scale] "
    "[--pretrain none|cross_patient] [--sensors full|headset|eeg_acc|eeg_only]"
)


def parse_args(argv):
    if len(argv) < 4:
        raise ValueError(USAGE)
    patient_id = argv[1]
    graph_strategy = argv[2]
    try:
        sop_minutes = int(argv[3])
    except ValueError:
        raise ValueError(f"sop_minutes has to be a whole number, we got {argv[3]!r}")
    if sop_minutes not in ALLOWED_SOP_MINUTES:
        raise ValueError(
            "sop_minutes has to be 30, 15, 10, or 5, "
            f"we got {sop_minutes!r}"
        )
    picked = {}
    rest = list(argv[4:])
    while len(rest) > 0:
        flag = rest.pop(0)
        if flag not in FLAG_NAMES:
            raise ValueError(f"unknown flag {flag!r}, " + USAGE)
        if len(rest) < 1:
            raise ValueError(f"flag {flag} needs a value, " + USAGE)
        key = FLAG_NAMES[flag]
        if key in picked:
            raise ValueError(f"flag {flag} given twice, " + USAGE)
        picked[key] = rest.pop(0)
    recipe = normalize_recipe(
        loss=picked.get("loss"),
        augmentation=picked.get("augmentation"),
        pretrain=picked.get("pretrain"),
        sensors=picked.get("sensors"),
    )
    leaf = recipe_leaf(recipe)
    return patient_id, graph_strategy, sop_minutes, recipe, leaf


def command_string(patient_id, graph_strategy, sop_minutes, recipe):
    command = (
        f"python train_variant.py {patient_id} {graph_strategy} {sop_minutes} "
        f"--loss {recipe['loss']} --augment {recipe['augmentation']} "
        f"--pretrain {recipe['pretrain']}"
    )
    if recipe["sensors"] != "full":
        command += f" --sensors {recipe['sensors']}"
    return command


def main():
    if not torch.cuda.is_available():
        raise ValueError("need a gpu for this, we got cpu")
    patient_id, graph_strategy, sop_minutes, recipe, leaf = parse_args(sys.argv)
    # mamba_ssm is only installed on the gpu pod
    import mamba_ssm

    install_logged_train_loop()
    device = "cuda"
    here = os.path.dirname(os.path.abspath(__file__))
    compile_start = time.perf_counter()
    warm = load_evobrain(device=device)
    warm = warm.to(device)
    warm_optimizer = make_optimizer(warm)
    input_seq, seq_lengths, adj, labels = make_batch(LOCKED["train_batch_size"], device)
    one_train_step(warm, warm_optimizer, input_seq, seq_lengths, adj, labels, 1.0)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start
    del warm, warm_optimizer, input_seq, seq_lengths, adj, labels

    row = {
        "patient_id": patient_id,
        "graph_strategy": graph_strategy,
        "sop_minutes": sop_minutes,
    }
    row.update(recipe_extra(recipe))
    try:
        row.update(
            _train_variant(
                here,
                patient_id,
                graph_strategy,
                sop_minutes,
                recipe,
                leaf,
                device,
                compile_seconds,
            )
        )
        row["status"] = "done"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        row["traceback"] = traceback.format_exc()
    print(json.dumps(row, indent=2))
    if row["status"] != "done":
        raise SystemExit(1)


def _train_variant(
    here, patient_id, graph_strategy, sop_minutes, recipe, leaf, device, compile_seconds
):
    import mamba_ssm

    source = _ready_folder(here, patient_id, graph_strategy, sop_minutes)
    command = command_string(patient_id, graph_strategy, sop_minutes, recipe)
    folder = prepare_variant_folder(
        here,
        patient_id,
        source,
        recipe,
        leaf,
        graph_strategy,
        sop_minutes,
        command,
    )
    if graph_strategy == "attention":
        model = load_attention_evobrain(device=device)
    else:
        model = load_evobrain(device=device)
    model = model.to(device)
    loss_fn = make_loss_fn(recipe["loss"])
    augment_fn = make_augment_fn(recipe["augmentation"], device=device)
    sensor_mask_fn = make_sensor_mask_fn(recipe["sensors"])
    extra = {
        "train_model": "evobrain" if graph_strategy != "attention" else "evobrain attention",
        "sop_minutes": sop_minutes,
        "train_torch_version": torch.__version__,
        "command": command,
        "seed": LOCKED["seed"],
        "dataset_accession": LOCKED["dataset_accession"],
        "dataset_version": LOCKED["dataset_version"],
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "cuda_version": torch.version.cuda,
        "mamba_ssm_version": mamba_ssm.__version__,
        "train_batch_size": LOCKED["train_batch_size"],
        "eval_batch_size": LOCKED["eval_batch_size"],
        "graph_strategy": graph_strategy,
    }
    extra.update(recipe_extra(recipe))
    if recipe["pretrain"] == "cross_patient":
        pretrain_dir = os.path.join(folder, "pretrain", "checkpoints")
        info = pretrain_model(
            here,
            model,
            patient_id,
            graph_strategy,
            sop_minutes,
            pretrain_dir,
            loss_fn=loss_fn,
            augment_fn=augment_fn,
            sensor_mask_fn=sensor_mask_fn,
            device=device,
        )
        extra.update(info)
        extra["pretrain_checkpoint"] = os.path.join(pretrain_dir, "best.pth.tar")
    optimizer = make_optimizer(model)
    _folder, _metrics = train_packed_eval_run(
        folder,
        model,
        optimizer,
        extra=extra,
        device=device,
        compile_seconds=compile_seconds,
        replace_existing_train=True,
        loss_fn=loss_fn,
        augment_fn=augment_fn,
        sensor_mask_fn=sensor_mask_fn,
    )
    _save_train_scores(folder, model, device, sensor_mask_fn=sensor_mask_fn)
    replace_metrics_keys(folder, _epoch_keys(folder))
    metrics = read_metrics(folder)
    return {
        "run_folder": folder,
        "best_epoch": metrics.get("best_epoch"),
        "n_epochs_run": metrics.get("n_epochs_run"),
        "best_monitor_bce": metrics.get("best_monitor_bce"),
        "logit_zero_monitor_bce": metrics.get("logit_zero_monitor_bce"),
        "train_seconds": metrics.get("train_seconds"),
        "score_seconds": metrics.get("score_seconds"),
        "pos_weight": metrics.get("pos_weight"),
        "parameter_count": metrics.get("parameter_count"),
    }


if __name__ == "__main__":
    main()
