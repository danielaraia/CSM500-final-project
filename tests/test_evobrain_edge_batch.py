import torch

from EvoBrain.model.edge_batch import (
    canonicalize_eigenvector_signs,
    time_major_edge_features,
)


def test_graph_batch_keeps_each_sample_separate():
    first = torch.arange(12).reshape(1, 3, 2, 2)
    second = first + 100
    result = time_major_edge_features(torch.cat((first, second)))
    torch.testing.assert_close(result[:, 0], first[0].reshape(3, 4, 1))
    torch.testing.assert_close(result[:, 1], second[0].reshape(3, 4, 1))


def test_eigenvector_sign_is_repeatable():
    vectors = torch.tensor([[-0.2, 0.1], [0.8, -0.9]])
    torch.testing.assert_close(
        canonicalize_eigenvector_signs(vectors),
        canonicalize_eigenvector_signs(-vectors),
    )
