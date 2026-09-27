import torch


class TinyLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(1, 1, bias=True)

    def forward(self, input_seq, seq_lengths, adj):
        # mean of the whole window. this is a laptop stand in, not evobrain
        pooled = input_seq.mean(dim=(1, 2, 3)).unsqueeze(-1)
        return self.w(pooled)
