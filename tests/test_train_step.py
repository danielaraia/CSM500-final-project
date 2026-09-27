import math

import pytest
import torch

from model.train_step import make_optimizer, one_train_step


class _TinyLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(1, 1, bias=True)

    def forward(self, input_seq, seq_lengths, adj):
        pooled = input_seq.mean(dim=(1, 2, 3)).unsqueeze(-1)
        return self.w(pooled)


def test_one_step_gives_a_finite_loss_and_moves_weights():
    torch.manual_seed(123)
    model = _TinyLogits()
    optimizer = make_optimizer(model)
    batch = 8
    input_seq = torch.randn(batch, 60, 8, 128)
    seq_lengths = torch.full((batch,), 60)
    adj = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(batch, 60, 1, 1)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
    before = model.w.weight.detach().clone()
    loss = one_train_step(model, optimizer, input_seq, seq_lengths, adj, labels, 7.0)
    after = model.w.weight.detach()
    assert math.isfinite(loss)
    assert loss > 0.0
    assert not torch.equal(before, after)


def test_bad_pos_weight_fails():
    torch.manual_seed(123)
    model = _TinyLogits()
    optimizer = make_optimizer(model)
    input_seq = torch.zeros(2, 60, 8, 128)
    seq_lengths = torch.tensor([60, 60])
    adj = torch.zeros(2, 60, 8, 8)
    labels = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="we got"):
        one_train_step(model, optimizer, input_seq, seq_lengths, adj, labels, 0.0)
