import math

import pytest
import torch

from augment import JITTER_SIGMA, SCALE_HIGH, SCALE_LOW, jitter_scale, make_augment_fn
from core.config import LOCKED
from tiny_model import TinyLogits
from train_loop import train_until_stop
from train_step import make_optimizer


def test_jitter_scale_is_deterministic_under_seed():
    torch.manual_seed(123)
    features = torch.randn(6, 60, 8, 128)
    gen_a = torch.Generator().manual_seed(LOCKED["seed"])
    gen_b = torch.Generator().manual_seed(LOCKED["seed"])
    out_a = jitter_scale(features, gen_a)
    out_b = jitter_scale(features, gen_b)
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, features)


def test_jitter_scale_preserves_shapes_and_leaves_input_untouched():
    torch.manual_seed(123)
    features = torch.randn(6, 60, 8, 128)
    before = features.clone()
    out = jitter_scale(features, torch.Generator().manual_seed(1))
    assert out.shape == features.shape
    assert out.dtype == features.dtype
    assert torch.equal(features, before)


def test_jitter_scale_stays_near_the_input():
    torch.manual_seed(123)
    features = torch.randn(4, 60, 8, 128)
    out = jitter_scale(features, torch.Generator().manual_seed(7))
    # scale is inside 0.9 to 1.1 and jitter sigma is small
    assert SCALE_LOW == 0.9
    assert SCALE_HIGH == 1.1
    assert JITTER_SIGMA <= 0.05
    assert float((out - features).abs().max()) < 1.0


def test_make_augment_fn_plumbing():
    assert make_augment_fn("none") is None
    with pytest.raises(ValueError, match="augment has to be"):
        make_augment_fn("smote")
    fn = make_augment_fn("jitter_scale")
    other = make_augment_fn("jitter_scale")
    features = torch.randn(4, 60, 8, 128)
    # both fns use the fixed seed, so they produce the same stream
    assert torch.equal(fn(features), other(features))


def test_train_never_augments_the_monitor_slice():
    torch.manual_seed(123)
    model = TinyLogits()
    optimizer = make_optimizer(model)
    train = _tiny_set(8, fill=1.0)
    monitor = _tiny_set(8, fill=-1.0)
    seen = []
    real = make_augment_fn("jitter_scale")

    def recording(features):
        seen.append(float(features.mean()))
        return real(features)

    train_until_stop(
        model,
        optimizer,
        train[0],
        train[1],
        train[2],
        monitor[0],
        monitor[1],
        monitor[2],
        pos_weight=7.0,
        max_epochs=2,
        patience=5,
        train_batch_size=8,
        augment_fn=recording,
    )
    # one train batch per epoch, and every augmented batch is train data
    assert len(seen) == 2
    for value in seen:
        assert value > 0.0


def test_zero_lr_stops_the_same_with_augmentation():
    torch.manual_seed(123)
    model = TinyLogits()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    train = _tiny_set(8)
    monitor = _tiny_set(8)
    best_epoch, best_loss, n_epochs_run = train_until_stop(
        model,
        optimizer,
        train[0],
        train[1],
        train[2],
        monitor[0],
        monitor[1],
        monitor[2],
        pos_weight=7.0,
        max_epochs=20,
        patience=2,
        train_batch_size=8,
        augment_fn=make_augment_fn("jitter_scale"),
    )
    assert best_epoch == 1
    assert n_epochs_run == 3
    assert math.isfinite(best_loss)


def _tiny_set(n_windows, fill=0.0):
    if fill == 0.0:
        features = torch.randn(n_windows, 60, 8, 128)
    else:
        features = torch.full((n_windows, 60, 8, 128), fill)
    adjacency = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(n_windows, 60, 1, 1)
    labels = torch.zeros(n_windows)
    labels[-1] = 1.0
    return features, adjacency, labels
