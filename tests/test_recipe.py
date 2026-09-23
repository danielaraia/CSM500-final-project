import pytest

from recipe import (
    DEFAULT_RECIPE,
    is_default,
    normalize_recipe,
    recipe_extra,
    recipe_leaf,
)


def test_default_recipe_has_no_variant_leaf():
    recipe = normalize_recipe()
    assert recipe == DEFAULT_RECIPE
    assert is_default(recipe)
    with pytest.raises(ValueError, match="train_sop.py"):
        recipe_leaf(recipe)


def test_recipe_leaf_names_each_lever():
    assert recipe_leaf(normalize_recipe(loss="focal")) == "eval1-focal"
    assert recipe_leaf(normalize_recipe(augmentation="jitter_scale")) == "eval1-aug"
    assert recipe_leaf(normalize_recipe(pretrain="cross_patient")) == "eval1-pre"
    recipe = normalize_recipe(
        loss="focal", augmentation="jitter_scale", pretrain="cross_patient"
    )
    assert recipe_leaf(recipe) == "eval1-focal-aug-pre"
    assert not is_default(recipe)


def test_recipe_rejects_unknown_values():
    with pytest.raises(ValueError, match="loss has to be"):
        normalize_recipe(loss="cosine")
    with pytest.raises(ValueError, match="augmentation has to be"):
        normalize_recipe(augmentation="smote")
    with pytest.raises(ValueError, match="pretrain has to be"):
        normalize_recipe(pretrain="self")


def test_recipe_extra_records_the_identity():
    extra = recipe_extra(normalize_recipe(loss="focal"))
    assert extra["loss"] == "focal"
    assert extra["augmentation"] == "none"
    assert extra["pretrain"] == "none"
    assert extra["recipe"] == "focal+none+none"
