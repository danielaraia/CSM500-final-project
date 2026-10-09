import numpy as np
import pytest
import torch

from evaluation.experiment_analysis import interval
from graphs.identity import IdentityGraphBackbone
from train.train_logistic import summary_features


class CapturesGraph(torch.nn.Module):
    def forward(self, features, lengths, graph):
        self.graph = graph
        return graph.mean(dim=(1, 2, 3))


def test_identity_graph_has_no_links_between_nodes():
    base = CapturesGraph()
    model = IdentityGraphBackbone(base)
    features = torch.zeros(2, 3, 8, 4)
    model(features, None, torch.randn(2, 3, 8, 8))
    expected = torch.eye(8).view(1, 1, 8, 8).expand(2, 3, 8, 8)
    torch.testing.assert_close(base.graph, expected)


def test_logistic_features_average_time():
    features = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    result = summary_features(features)
    assert result.shape == (2, 20)
    np.testing.assert_array_equal(
        result[0], features[0].mean(axis=0).reshape(-1)
    )
    assert summary_features(np.zeros((0, 3, 4, 5))).shape == (0, 20)


def test_bootstrap_is_repeatable():
    first = interval([0.0, 0.5, 1.0])
    second = interval([0.0, 0.5, 1.0])
    assert first == second
    assert first["mean"] == pytest.approx(0.5)
