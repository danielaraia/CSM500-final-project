from data.augment import AUGMENT_NAMES
from focal_loss import LOSS_NAMES
from data.sensor_mask import SENSOR_TIERS

PRETRAIN_NAMES = ("none", "cross_patient")
DEFAULT_RECIPE = {
    "loss": "weighted_bce",
    "augmentation": "none",
    "pretrain": "none",
    "sensors": "full",
}


def normalize_recipe(loss=None, augmentation=None, pretrain=None, sensors=None):
    recipe = {
        "loss": loss if loss is not None else DEFAULT_RECIPE["loss"],
        "augmentation": (
            augmentation if augmentation is not None else DEFAULT_RECIPE["augmentation"]
        ),
        "pretrain": pretrain if pretrain is not None else DEFAULT_RECIPE["pretrain"],
        "sensors": sensors if sensors is not None else DEFAULT_RECIPE["sensors"],
    }
    if recipe["loss"] not in LOSS_NAMES:
        raise ValueError(
            "loss has to be weighted_bce or focal, "
            f"we got {recipe['loss']!r}"
        )
    if recipe["augmentation"] not in AUGMENT_NAMES:
        raise ValueError(
            "augmentation has to be none or jitter_scale, "
            f"we got {recipe['augmentation']!r}"
        )
    if recipe["pretrain"] not in PRETRAIN_NAMES:
        raise ValueError(
            "pretrain has to be none or cross_patient, "
            f"we got {recipe['pretrain']!r}"
        )
    if recipe["sensors"] not in SENSOR_TIERS:
        raise ValueError(
            "sensors has to be full, headset, eeg_acc, or eeg_only, "
            f"we got {recipe['sensors']!r}"
        )
    return recipe


def is_default(recipe):
    return recipe == DEFAULT_RECIPE


def recipe_leaf(recipe):
    # variant runs get their own leaf so they never write over the default run
    if is_default(recipe):
        raise ValueError(
            "all levers are default, run train_sop.py for the locked recipe"
        )
    leaf = "eval1"
    if recipe["loss"] == "focal":
        leaf += "-focal"
    if recipe["augmentation"] == "jitter_scale":
        leaf += "-aug"
    if recipe["pretrain"] == "cross_patient":
        leaf += "-pre"
    if recipe["sensors"] != DEFAULT_RECIPE["sensors"]:
        leaf += "-" + recipe["sensors"]
    return leaf


def recipe_extra(recipe):
    # the screening ledger reads these keys from the run folder config json
    extra = {
        "loss": recipe["loss"],
        "augmentation": recipe["augmentation"],
        "pretrain": recipe["pretrain"],
        "recipe": (
            recipe["loss"]
            + "+"
            + recipe["augmentation"]
            + "+"
            + recipe["pretrain"]
        ),
    }
    # full is the usual sensor set, so it adds nothing to the run identity
    if recipe["sensors"] != DEFAULT_RECIPE["sensors"]:
        extra["sensors"] = recipe["sensors"]
        extra["recipe"] += "+" + recipe["sensors"]
    return extra
