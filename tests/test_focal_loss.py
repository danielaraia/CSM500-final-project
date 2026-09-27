import math

import pytest
import torch
from torch.nn import BCEWithLogitsLoss

from model.focal_loss import FOCAL_GAMMA, focal_loss_with_logits, make_loss_fn
from model.tiny_model import TinyLogits
from model.train_step import make_optimizer, one_train_step


def test_focal_matches_hand_computed_values():
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 0.0])
    loss = focal_loss_with_logits(logits, labels, 2.0)
    # p is 0.5 everywhere: pos gets pos_weight * 0.5^2 * log 2, neg 0.5^2 * log 2
    expected = (2.0 * 0.25 * math.log(2.0) + 0.25 * math.log(2.0)) / 2.0
    assert float(loss) == pytest.approx(expected)


def test_focal_gamma_zero_is_the_locked_weighted_bce():
    torch.manual_seed(123)
    logits = torch.randn(9)
    labels = (torch.rand(9) > 0.5).float()
    focal = focal_loss_with_logits(logits, labels, 3.0, gamma=0.0)
    weight = torch.as_tensor(3.0)
    bce = BCEWithLogitsLoss(pos_weight=weight)(logits, labels)
    assert float(focal) == pytest.approx(float(bce))


def test_focal_downweights_easy_examples():
    logits = torch.tensor([10.0, -10.0])
    labels = torch.tensor([1.0, 0.0])
    focal = focal_loss_with_logits(logits, labels, 2.0)
    bce = BCEWithLogitsLoss(pos_weight=torch.as_tensor(2.0))(logits, labels)
    assert float(focal) < float(bce)


def test_focal_rejects_bad_arguments():
    with pytest.raises(ValueError, match="pos_weight"):
        focal_loss_with_logits(torch.zeros(2), torch.zeros(2), 0.0)
    with pytest.raises(ValueError, match="gamma"):
        focal_loss_with_logits(torch.zeros(2), torch.zeros(2), 1.0, gamma=-1.0)
    with pytest.raises(ValueError, match="same batch"):
        focal_loss_with_logits(torch.zeros(3), torch.zeros(2), 1.0)


def test_make_loss_fn_plumbing():
    assert make_loss_fn("weighted_bce") is None
    fn = make_loss_fn("focal")
    loss = fn(torch.tensor([0.0]), torch.tensor([1.0]), 2.0)
    expected = focal_loss_with_logits(
        torch.tensor([0.0]), torch.tensor([1.0]), 2.0, gamma=FOCAL_GAMMA
    )
    assert float(loss) == pytest.approx(float(expected))
    with pytest.raises(ValueError, match="loss has to be"):
        make_loss_fn("cosine")


def test_one_train_step_default_matches_explicit_weighted_bce():
    torch.manual_seed(123)
    model_a = TinyLogits()
    torch.manual_seed(123)
    model_b = TinyLogits()
    features = torch.randn(8, 60, 8, 128)
    adj = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(8, 60, 1, 1)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
    seq = torch.full((8,), 60)

    def locked_bce(logits, labs, pos_weight):
        weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        return BCEWithLogitsLoss(pos_weight=weight)(logits, labs)

    loss_a = one_train_step(
        model_a, make_optimizer(model_a), features, seq, adj, labels, 7.0
    )
    loss_b = one_train_step(
        model_b, make_optimizer(model_b), features, seq, adj, labels, 7.0,
        loss_fn=locked_bce,
    )
    assert loss_a == pytest.approx(loss_b)
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(param_a.detach(), param_b.detach())


def test_one_train_step_with_focal_moves_weights():
    torch.manual_seed(123)
    model = TinyLogits()
    optimizer = make_optimizer(model)
    features = torch.randn(8, 60, 8, 128)
    adj = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(8, 60, 1, 1)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
    seq = torch.full((8,), 60)
    before = model.w.weight.detach().clone()
    loss = one_train_step(
        model, optimizer, features, seq, adj, labels, 7.0,
        loss_fn=make_loss_fn("focal"),
    )
    after = model.w.weight.detach()
    assert math.isfinite(loss)
    assert loss > 0.0
    assert not torch.equal(before, after)
