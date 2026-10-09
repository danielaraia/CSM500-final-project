import torch
import torch.nn as nn

from model.evobrain_load import load_evobrain


class ZeroLaplacianPE:
    def __init__(self, n_eigenvectors):
        self.n_eigenvectors = n_eigenvectors

    def __call__(self, data):
        data.laplacian_eigenvector_pe = torch.zeros(
            (data.x.shape[0], self.n_eigenvectors),
            dtype=data.x.dtype,
            device=data.x.device,
        )
        return data


class IdentityGraphBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_seq, seq_lengths, adj):
        batch, steps, nodes, _ = input_seq.shape
        identity = torch.eye(
            nodes, dtype=adj.dtype, device=input_seq.device
        )
        identity = identity.view(1, 1, nodes, nodes).expand(
            batch, steps, nodes, nodes
        )
        return self.backbone(input_seq, seq_lengths, identity)


def load_identity_evobrain(device="cpu"):
    backbone = load_evobrain(device=device)
    n_eigenvectors = backbone.evobrain.num_eigenvectors
    backbone.evobrain.laplacian_pe = ZeroLaplacianPE(n_eigenvectors)
    return IdentityGraphBackbone(backbone)
